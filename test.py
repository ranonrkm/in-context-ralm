from typing import List
from functools import partial
import numpy as np
import torch
import faiss
from datasets import load_dataset, load_from_disk
from transformers import AutoTokenizer, AutoModel

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

query_enc="facebook/dragon-plus-query-encoder"
device="cuda:0"
encoder = AutoEmbeddingModel(query_enc, device, pooling_strategy='cls')

datastore = load_dataset("wikipedia", "20220301.en", split="train", cache_dir="/data/user_data/rsadhukh/cache")
datastore = datastore.map(partial(split_documents, n=200), 
                                    batched=True, num_proc=8, remove_columns=datastore.column_names)

prefixed_dataset = load_from_disk("/data/user_data/rsadhukh/wikipedia/facebook/dragon-plus-context-encoder/prefixed_query_annotated")

# index = faiss.read_index('/data/user_data/rsadhukh/wikipedia/facebook/dragon-plus-context-encoder/index_OPQ64_256_IVF4096_PQ64.indexed')
# index.nprobe = 128

index_dir='/data/user_data/rsadhukh/wikipedia/facebook/dragon-plus-context-encoder'
indexes = [faiss.read_index(f"{index_dir}/index_{i}.index") for i in range(8)]
index = MultiIndex(index_dir)
print("ntotal", index.ntotal)

query = prefixed_dataset[:3]['prefix']
embs = encoder(query).cpu().numpy()
D, I = index.search(embs, 10)
texts = []
import pdb; pdb.set_trace()
for i in range(len(query)):
    print(f"Query: {query[i]}")
    for j in range(10):
        print(f"Rank {j+1}: {datastore[I[i][j]]['text']}")
    print("\n")

