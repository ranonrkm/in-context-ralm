import copy
from typing import List
import numpy as np
import torch
import torch.nn.functional as F
import faiss
import faiss.contrib.torch_utils

from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer
from huggingface_hub import login


def load_tokenizer(model_name):
    if "llama" in model_name:
        return LlamaTokenizer.from_pretrained(model_name)
    return AutoTokenizer.from_pretrained(model_name)


def load_model_and_tokenizer(model_name, model_parallelism=False, cache_dir=None, auth_token=None):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    device_count = torch.cuda.device_count()

    config = AutoConfig.from_pretrained(model_name)
    model_args = {}
    if cache_dir is not None:
        model_args["cache_dir"] = cache_dir
    if model_parallelism:
        model_args["device_map"] = "auto"
        model_args["low_cpu_mem_usage"] = True
    if hasattr(config, "torch_dtype") and config.torch_dtype is not None:
        model_args["torch_dtype"] = config.torch_dtype
    if auth_token is not None:
        model_args["use_auth_token"] = auth_token

    model = AutoModelForCausalLM.from_pretrained(model_name, **model_args).eval()
    if not model_parallelism:
        model = model.to(device)
    tokenizer = load_tokenizer(model_name)

    if device_count > 1 and not model_parallelism:
        model = torch.nn.DataParallel(model)

    return model, tokenizer, config, device


class ActivationCapturer(torch.nn.Module):
    def __init__(self, layer, capture_input=False):
        super().__init__()
        self.layer = layer
        self.capture_input = capture_input
        self.captured = None

    def forward(self, module, input, output):
        if self.capture_input:
            self.captured = input[0].detach()
        else:
            if isinstance(output, tuple):
                self.captured = output[0].detach()
            else:
                self.captured = output.detach()

class KNNLM:
    def __init__(self, tokenizer, model, 
                 index, vals, k=1024, knn_lambda=0.25, device='cuda:0'):
        self.tokenizer = tokenizer
        self.model = model
        capture_layer = model.base_model.h[-1].mlp
        self.activation_capturer = ActivationCapturer(capture_layer, capture_input=True)
        capture_layer.register_forward_hook(self.activation_capturer)
        self.model.lm_head.register_forward_hook(self.post_fwd_hook)
        
        res = faiss.StandardGpuResources()
        res.setTempMemory(1024*1024*1024*4)
        co = faiss.GpuClonerOptions()
        co.useFloat16 = True
        self.knn_index = faiss.index_cpu_to_gpu(res, 0, index, co)
        self.vals = vals
        self.topk = k
        self.knn_lambda = knn_lambda
        self.labels = None
        self.device = device
        self.model.to(device)

    def knns_to_log_probs(self, ids, scores, temp=1.0):
        probs = F.softmax(scores / temp, dim=-1)
        vals = self.vals[ids.cpu().numpy()].squeeze(-1)
        knn_log_probs = torch.full(size=(vals.shape[:-1] + (self.model.config.vocab_size,)), fill_value=0.0, device=ids.device) \
            .scatter_add(-1, torch.from_numpy(vals).long().to(probs.device), probs).log()
        knn_log_probs = torch.nan_to_num(knn_log_probs, neginf=float("-inf"))
        return knn_log_probs, vals
    
    def post_fwd_hook(self, module, input, output):
        if self.knn_lambda < 1e-4:
            return output
        batch, time_dim, vocab_size = output.shape
        shift = 1
        lm_logits = output
        lm_log_probs = F.log_softmax(lm_logits, dim=-1)
        
        if self.labels is not None:
            nonpad_mask = torch.cat([
                self.labels[:, shift:] != -100, 
                torch.zeros([self.labels.shape[0], shift], dtype=torch.bool).to(output.device)
            ], axis=-1)
        else:
            nonpad_mask = torch.cat([
                    torch.zeros([batch, time_dim - 1], dtype=torch.bool),
                    torch.ones([batch, 1], dtype=torch.bool),
                ], axis=-1).to(output.device)

        lm_log_probs = lm_log_probs[nonpad_mask]

        queries = self.activation_capturer.captured # [batch, time_dim, dim]
        queries = queries[nonpad_mask]

        D, I = self.knn_index.search(queries.view(-1, queries.shape[-1]).float(), self.topk)
        if self.knn_index.metric_type == 1:
            D = -1. * D

        knn_log_probs, _ = self.knns_to_log_probs(I, D, temp=1.0)
        log_probs = torch.logaddexp(lm_log_probs + np.log(1 - self.knn_lambda),
                                knn_log_probs + np.log(self.knn_lambda))
        
        output[nonpad_mask] = log_probs.to(output.dtype)
        return output
    
    def __call__(self, input_ids, labels=None):
        self.labels = labels
        return self.model(input_ids, labels=labels)
    

def load_knnlm_and_tokenizer(model_name,
                             index_path, vals_path, k=1024, knn_lambda=0.25, 
                             model_parallelism=False, cache_dir=None, auth_token=None):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    device_count = torch.cuda.device_count()

    config = AutoConfig.from_pretrained(model_name)
    model_args = {}
    if cache_dir is not None:
        model_args["cache_dir"] = cache_dir
    if model_parallelism:
        model_args["device_map"] = "auto"
        model_args["low_cpu_mem_usage"] = True
    if hasattr(config, "torch_dtype") and config.torch_dtype is not None:
        model_args["torch_dtype"] = config.torch_dtype
    if auth_token is not None:
        model_args["use_auth_token"] = auth_token

    model = AutoModelForCausalLM.from_pretrained(model_name, **model_args).eval()
    if not model_parallelism:
        model = model.to(device)
    tokenizer = load_tokenizer(model_name)

    if device_count > 1 and not model_parallelism:
        model = torch.nn.DataParallel(model)

    index = faiss.read_index(index_path)
    vals = np.memmap(vals_path, dtype=np.int32, mode='r', shape=(index.ntotal, 1))

    knnlm = KNNLM(tokenizer, model, index, vals, k=k, knn_lambda=knn_lambda, device=device)

    return knnlm, tokenizer, config, device


class TTTLM:
    def __init__(self, tokenizer, model, 
                 tr_batch_size=1, max_tr_niters=10, 
                 lr=5e-6, adam_epsilon=1e-8, device='cuda:0'):
        self.tokenizer = tokenizer
        self.model = model
        self.device = device
        self.block_size = model.config.max_position_embeddings
        self.stride = self.block_size // 2
        self._orig_state = copy.deepcopy(model.state_dict())
        self.tr_batch_size = tr_batch_size
        self.max_tr_niters = max_tr_niters
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=lr, eps=adam_epsilon
        )
        self.model.to(device)

    def _reset_model(self):
        self.model.load_state_dict(self._orig_state)

    def finetune(self, train_input_ids):
        train_input_ids, train_labels = self.group_texts(train_input_ids)
        self.model.train()
        niter = 0
        begin = 0
        while niter < self.max_tr_niters:
            end = min(begin + self.tr_batch_size, len(train_input_ids))
            self.optimizer.zero_grad()
            input_ids_batch = train_input_ids[begin : end].to(self.device)
            labels_batch = train_labels[begin : end].to(self.device)
            outputs = self.model(input_ids_batch, labels=labels_batch)
            loss = outputs.loss
            loss.backward()
            self.optimizer.step()
            niter += 1
            begin = end
            if begin >= len(train_input_ids):
                begin = 0
        self.model.eval()

    def group_texts(self, all_token_ids: List[int]):
        input_ids = []
        labels = []
        padding_index = -100
        total_length = len(all_token_ids)
        if total_length >= self.block_size:
            total_length = (total_length // self.block_size) * self.block_size
        for i in range(0, total_length, self.stride):
            begin_loc = max(i + self.stride - self.block_size, 0)
            end_loc = min(i + self.stride, total_length)
            trg_len = end_loc - i
            cur_input_ids = all_token_ids[begin_loc:end_loc]
            cur_labels = list(cur_input_ids)
            cur_labels[:-trg_len] = [padding_index] * (len(cur_input_ids) - trg_len)

            if len(cur_input_ids) < self.block_size:
                padding_size = self.block_size - len(cur_input_ids)
                pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
                cur_input_ids += [pad_token_id] * padding_size
                cur_labels += [padding_index] * padding_size
            
            input_ids.append(cur_input_ids)
            labels.append(cur_labels)

            return torch.LongTensor(input_ids), torch.LongTensor(labels)

    def __call__(self, input_ids, labels=None):
        res = self.model(input_ids, labels=labels)
        self._reset_model()
        return res
        
def load_tttlm_and_tokenizer(model_name, model_parallelism=False, cache_dir=None, auth_token=None):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    device_count = torch.cuda.device_count()

    config = AutoConfig.from_pretrained(model_name)
    model_args = {}
    if cache_dir is not None:
        model_args["cache_dir"] = cache_dir
    if model_parallelism:
        model_args["device_map"] = "auto"
        model_args["low_cpu_mem_usage"] = True
    if hasattr(config, "torch_dtype") and config.torch_dtype is not None:
        model_args["torch_dtype"] = config.torch_dtype
    if auth_token is not None:
        model_args["use_auth_token"] = auth_token

    model = AutoModelForCausalLM.from_pretrained(model_name, **model_args).eval()
    if not model_parallelism:
        model = model.to(device)
    tokenizer = load_tokenizer(model_name)

    if device_count > 1 and not model_parallelism:
        model = torch.nn.DataParallel(model)

    tttlm = TTTLM(tokenizer, model, device=device)

    return tttlm, tokenizer, config, device