from typing import List
from functools import partial
import numpy as np
import torch

import faiss
import faiss.contrib.torch_utils

from datasets import load_dataset, concatenate_datasets
from transformers import AutoTokenizer, AutoModel

from tqdm import tqdm

def unwind_index_ivf(index):
    if isinstance(index, faiss.IndexPreTransform):
        assert index.chain.size() == 1
        vt = index.chain.at(0)
        index_ivf, vt2 = unwind_index_ivf(faiss.downcast_index(index.index))
        assert vt2 is None
        if vt is None:
            vt = lambda x: x
        else:
            vt = faiss.downcast_VectorTransform(vt)
        return index_ivf, vt
    if hasattr(faiss, "IndexRefine") and isinstance(index, faiss.IndexRefine):
        return unwind_index_ivf(faiss.downcast_index(index.base_index))
    if isinstance(index, faiss.IndexIVF):
        return index, None
    else:
        return None, None


class AutoEmbeddingModel:
    def __init__(self, embedding_model_checkpoint, device, pooling_strategy='mean'):
        self.tokenizer = AutoTokenizer.from_pretrained(embedding_model_checkpoint)
        self.model = AutoModel.from_pretrained(embedding_model_checkpoint, 
                        cache_dir="/data/user_data/rsadhukh/cache").to(device)
        self.device = device
        self.pool_strategy = pooling_strategy

    def __call__(self, texts: List[str]) -> torch.Tensor:
        with torch.no_grad():
            tokens = self.tokenizer(texts, padding=True, truncation=True, return_tensors='pt', max_length=512)
            input_ids = tokens['input_ids'].to(self.device)
            attention_mask = tokens['attention_mask'].to(self.device)
            output = self.model(input_ids, attention_mask=attention_mask)
            if self.pool_strategy.lower() == 'mean':
                embs = output['last_hidden_state'] * attention_mask.unsqueeze(-1) # [batch_size, seq_len, emb_dim]
                embs = embs.sum(1) / attention_mask.sum(dim=-1, keepdim=True) # [batch_size, emb_dim]
            else:
                embs = output['last_hidden_state'][:, 0, :]
        return embs

def split_documents(documents: dict, n=200, text_col="text") -> dict:
    """Split documents into passages"""

    def split_text(text: str, n=200, character=" ") -> List[str]:
        """Split the text every ``n``-th occurrence of ``character``"""
        text = text.split(character)
        return [character.join(text[i : i + n]).strip() for i in range(0, len(text), n)]
    
    texts = []
    for text in documents[text_col]:
        if text is not None:
            for passage in split_text(text, n=n):
                if len(passage) > 10:
                    texts.append(passage)
    return {"text": texts}

class MultiIndex:
    def __init__(self, index_dir):
        self.indices = [faiss.read_index(f"{index_dir}/index_{i}.index") for i in range(8)]
        ntotals = [0] + [index.ntotal for index in self.indices][:-1]
        self.ntotals = np.cumsum(ntotals)
        self.ntotal = sum(ntotals)

    def search(self, query, k):
        scores = []
        ids = []
        for index_id, index in enumerate(self.indices):
            s, i = index.search(query, k)
            scores.append(s)
            ids.append(i + self.ntotals[index_id])
        scores = np.concatenate(scores, axis=1)
        ids = np.concatenate(ids, axis=1)
        # rerank based on scores
        sorted_indices = np.argsort(scores, axis=1)[:, ::-1]
        scores = np.take_along_axis(scores, sorted_indices, axis=1)
        ids = np.take_along_axis(ids, sorted_indices, axis=1)
        return scores[:, :k], ids[:, :k]


class DenseRetriever:
    def __init__(self, query_enc, tokenizer, num_tokens_for_query, 
                 dataset_name, index_path, text_col="text",
                 pooling_strategy='mean', device='cuda',
                 nprobe=64, gpu_index=False):
        self.encoder = AutoEmbeddingModel(query_enc, device, pooling_strategy=pooling_strategy)
        self.lm_tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        
        self.index = faiss.read_index(index_path)
        print("ntotal", self.index.ntotal)
        index_ivf, vt = unwind_index_ivf(self.index)
        if index_ivf is not None:
            index_ivf.nprobe = nprobe   

        if gpu_index:
            co = faiss.GpuMultipleClonerOptions()
            co.shard = True
            co.useFloat16 = True
            self.index = faiss.index_cpu_to_all_gpus(self.index, co)
        
        # self.index = MultiIndex(index_path)

        dataset_name, dataset_version = dataset_name.split(":")
        self.dataset = load_dataset(dataset_name, dataset_version, split="train", 
                                    cache_dir='/data/user_data/rsadhukh/cache')
        print("splitting docs...")
        datashards = []
        for i in range(8):
            datashard = self.dataset.shard(index=i, num_shards=8)
            datashard = datashard.map(partial(split_documents, n=200),
                                        batched=True, num_proc=8, remove_columns=self.dataset.column_names)
            datashards.append(datashard)
        self.dataset = concatenate_datasets(datashards)
        print("done splitting docs")
        
        assert self.index.ntotal == len(self.dataset), f"Index size {self.index.ntotal} != dataset size {len(self.dataset)}"

        self.num_tokens_for_query = num_tokens_for_query
        
    
    def _get_query_string(self, sequence_input_ids, target_begin_location, target_end_location, title=None):
        # We isolate the prefix to make sure that we don't take tokens from the future:
        prefix_tokens = sequence_input_ids[0, :target_begin_location]
        query_tokens = prefix_tokens[-self.num_tokens_for_query:]
        query_str = self.lm_tokenizer.decode(query_tokens)
        return query_str


    def retrieve(self, sequence_input_ids, dataset, k=1, batch_size=128):
        queries = [
            self._get_query_string(
                sequence_input_ids,
                d["begin_location"],
                d["end_location"],
                d["title"] if "title" in d else None
            )
            for d in dataset
        ]
        assert len(queries) == len(dataset)

        for begin in tqdm(range(0, len(queries), batch_size)):
            end = min(begin + batch_size, len(queries))
            embs = self.encoder(queries[begin :  end]).cpu().float().numpy()
            scores, ids = self.index.search(embs, k)
            cur_texts = self.dataset.select(ids.flatten())['text']
            for qid in range(begin, end):
                d = dataset[qid]
                d["query"] = queries[qid]
                texts_j = cur_texts[qid * k : (qid + 1) * k]
                scores_j = scores[qid - begin] 
                d["retrieved_docs"] = [{"text": text, "score": float(score)} for text, score in zip(texts_j, scores_j)]
        return dataset

            
        


