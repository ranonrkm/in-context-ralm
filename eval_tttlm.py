import os
import argparse
import json
import pickle

import numpy as np
import torch
import transformers
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
from datasets import load_dataset

from ralm.file_utils import print_args
from ralm.model_utils import load_tttlm_and_tokenizer

def evaluate_logprob_with_retrieved_docs(
        model,
        tokenizer,
        device,
        encodings,
        begin_loc,
        end_loc,
        trg_len,
        retrieved_item,
        num_docs=10
):

    input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)

    num_docs = len(retrieved_item["retrieved_docs"])

    retrieved_texts = []
    for doc_id in range(num_docs):
        retrieved_example = retrieved_item["retrieved_docs"][doc_id]

        doc_title = retrieved_example["title"] if "title" in retrieved_example else None
        doc_text = retrieved_example["text"]
        if doc_title:
            doc_text = doc_title + "\n" + doc_text
        retrieved_texts.append(doc_text)

    # concat the retrieved texts
    retrieved_texts = " ".join(retrieved_texts)
    encoded_retrieved_text = tokenizer(retrieved_texts)["input_ids"]

    labels = input_ids.clone()[0, -trg_len:]    
    loss_fct = CrossEntropyLoss(reduction="none")

    with torch.no_grad():
        pre_lm_logits= model(input_ids).logits[0, -trg_len-1:-1, :]
        pre_token_ppls = loss_fct(pre_lm_logits, labels).cpu()
        
    model.finetune(encoded_retrieved_text)

    with torch.no_grad():
        post_lm_logits = model(input_ids).logits[0, -trg_len-1:-1, :]
        post_token_ppls = loss_fct(post_lm_logits, labels).cpu()

    tokens_to_predict = labels.view(-1).cpu().tolist()

    return pre_token_ppls.sum(), post_token_ppls.sum(), \
            pre_token_ppls.tolist(), post_token_ppls.tolist(), tokens_to_predict


def eval_dataset(
        model,
        tokenizer,
        dataset,
        device,
        max_length,
        output_dir=None,
        stride=4,
        normalization_level="word",
        retrieval_dataset=None,
        retrieval_max_length=256,
        num_docs=10
):

    encodings = tokenizer(dataset, add_special_tokens=False, return_tensors="pt")

    print("Max context length:", max_length)
    # Number of tokens in dataset
    dataset_len = encodings.input_ids.size(1)
    print("Dataset length:", dataset_len)

    if normalization_level == "word":
        counter = dataset.count(" ")
    elif normalization_level == "token":
        counter = dataset_len
    else:
        raise ValueError(f"Unknown normalization_level: '{normalization_level}'")

    print("Normalization factor (num tokens/words..):", counter)

    prev_end_loc = 0

    idx = 0
    pre_nlls = []
    post_nlls = []
    all_pre_token_ppls = []
    all_post_token_ppls = [] 
    all_tokens_to_predict = []
    num_inputs_no_retrieval = 0
    running_tokens_to_predict = 0
    pbar = tqdm(range(0, dataset_len, stride))
    for begin_loc in pbar:
        end_loc = min(begin_loc + max_length, dataset_len)
        trg_len = end_loc - prev_end_loc  # may be different from stride on last loop
        if idx > 0 and retrieval_dataset is not None and len(retrieval_dataset[idx]["retrieved_docs"]) > 0:
            retrieved_example = retrieval_dataset[idx]
            assert retrieved_example["begin_location"] == prev_end_loc
            assert retrieved_example["end_location"] == end_loc

            pre_nll, post_nll, pre_token_ppls, post_token_ppls, tokens_to_predict = evaluate_logprob_with_retrieved_docs(
                model, tokenizer, device, encodings, begin_loc, end_loc, trg_len, retrieved_example,
            )

            # ****** TODO: to be moved **********
            pre_nlls.append(pre_nll)
            post_nlls.append(post_nll)
            all_pre_token_ppls.append(pre_token_ppls)
            all_post_token_ppls.append(post_token_ppls)
            all_tokens_to_predict.append(tokens_to_predict)
            assert len(all_pre_token_ppls) == len(all_tokens_to_predict)

            running_tokens_to_predict += trg_len
            running_pre_ppl = torch.exp(torch.stack(pre_nlls).sum() / running_tokens_to_predict).item()
            running_post_ppl = torch.exp(torch.stack(post_nlls).sum() / running_tokens_to_predict).item()
            pbar.set_description(f"Pre TTT PPL: {running_pre_ppl:.2f}, Post TTT PPL: {running_post_ppl:.2f}, no retrieval: {num_inputs_no_retrieval}")
        else:
            input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
            target_ids = input_ids.clone()
            target_ids[:, :-trg_len] = -100

            with torch.no_grad():
                outputs = model(input_ids, labels=target_ids)

                # Calculate per-token loss
                if trg_len < max_length:
                    neg_log_likelihood = outputs.loss * trg_len
                    lm_logits = outputs.logits[..., -trg_len-1:-1, :]
                    labels = target_ids[..., -trg_len:]
                else:
                    neg_log_likelihood = outputs.loss * (max_length - 1)
                    lm_logits = outputs.logits[..., :-1, :]
                    labels = target_ids[..., 1:]
                neg_log_likelihood = neg_log_likelihood.to(torch.float32).squeeze().cpu()
                lm_logits = lm_logits.to(torch.float32)

                loss_fct = CrossEntropyLoss(reduction="none")
                loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1)).cpu()
                token_ppls = loss.tolist()
                tokens_to_predict = labels.view(-1).cpu().tolist()

                pre_nll = post_nll = neg_log_likelihood
                pre_token_ppls = post_token_ppls = token_ppls
            
            num_inputs_no_retrieval += 1

        prev_end_loc = end_loc
        idx += 1
        if end_loc == dataset_len:
            break

    assert retrieval_dataset is None or len(retrieval_dataset) == idx
    
    # TODO: to be removed
    counter = running_tokens_to_predict
    pre_ppl = torch.exp(torch.stack(pre_nlls).sum() / counter).item()
    print("Pre TTT Perplexity:", pre_ppl)
    post_ppl = torch.exp(torch.stack(post_nlls).sum() / counter).item()
    print("Post TTT Perplexity:", post_ppl)
    
    if output_dir is not None:
        d = {"eval_pre_ppl": pre_ppl, "eval_post_ppl": post_ppl}
        if retrieval_dataset is not None:
            d["num_input_no_retrieval"] = num_inputs_no_retrieval
        with open(os.path.join(output_dir, "eval.json"), "w") as f:
            f.write(json.dumps(d) + "\n")

        with open(os.path.join(output_dir, "ppls.pkl"), "wb") as f:
            to_dump = (all_pre_token_ppls, all_post_token_ppls, all_tokens_to_predict)
            pickle.dump(to_dump, f)

    
def main(args):
    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)
    print_args(args, output_dir=args.output_dir)

    model, tokenizer, config, device = load_tttlm_and_tokenizer(args.model_name, model_parallelism=args.model_parallelism, cache_dir=args.cache_dir, auth_token=args.auth_token)

    # Model context size (e.g., 1024 for GPT-2)
    max_length = args.max_length
    model_max_length = config.n_positions if hasattr(config, "n_positions") else config.max_position_embeddings
    if max_length is None or max_length > model_max_length:
        max_length = model_max_length

    if args.load_from == "hf":
        dataset = load_dataset(args.dataset_path, args.dataset_name, split=args.dataset_split)
        dataset = "".join([x[args.text_col] if x[args.text_col] else " \n" for x in dataset])
    else:
        with open(args.dataset_path, "r") as f:
            dataset = f.read()

    transformers.logging.set_verbosity_error()
    retrieval_dataset = None
    if args.retrieved_file is not None:
        with open(args.retrieved_file, "r") as f:
            retrieval_dataset = json.load(f)

    eval_dataset(
        model,
        tokenizer,
        dataset,
        device,
        max_length=max_length,
        output_dir=args.output_dir,
        stride=args.stride,
        normalization_level=args.normalization_level,
        retrieval_dataset=retrieval_dataset,
        retrieval_max_length=args.retrieved_max_length,
        num_docs=10
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--output_dir", type=str)

    # Model params
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--max_length", type=int, default=None)
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--model_parallelism", action="store_true")
    parser.add_argument("--auth_token", type=str, default=None)

    # Dataset params
    parser.add_argument("--load_from", type=str, choices=["hf", "file"], default="hf")
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--dataset_name", type=str, default=None)
    parser.add_argument("--dataset_split", type=str, default="test")
    parser.add_argument("--text_col", type=str, default="text")
    parser.add_argument("--normalization_level", choices=["word", "token"], default="word")

    # retrieval params
    parser.add_argument("--retrieved_file", type=str, default=None)
    parser.add_argument("--retrieved_max_length", type=int, default=256)

    args = parser.parse_args()

    main(args)