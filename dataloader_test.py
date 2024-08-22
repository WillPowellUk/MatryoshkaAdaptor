from beir import util, LoggingHandler
from beir.datasets.data_loader import GenericDataLoader
import torch
from torch.utils.data import Dataset, DataLoader
import logging

# Set up logging
logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO, handlers=[LoggingHandler()])

# Define the dataset name and the path to store it
dataset = "nfcorpus"
url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset}.zip"
data_path = util.download_and_unzip(url, "datasets")

# Load train and test data
train_corpus, train_queries, train_qrels = GenericDataLoader(data_path).load(split="train")
dev_corpus, dev_queries, dev_qrels = GenericDataLoader(data_path).load(split="dev")
test_corpus, test_queries, test_qrels = GenericDataLoader(data_path).load(split="test")

class BEIRDataset(Dataset):
    def __init__(self, queries, corpus, qrels):
        self.queries = queries
        self.corpus = corpus
        self.qrels = qrels
        self.query_ids = list(queries.keys())
        
    def __len__(self):
        return len(self.query_ids)
    
    def __getitem__(self, idx):
        query_id = self.query_ids[idx]
        query_text = self.queries[query_id]
        
        relevant_docs = self.qrels.get(query_id, {})
        
        # Get all document ids and scores for this query
        doc_ids = list(relevant_docs.keys())
        scores = [relevant_docs[doc_id] for doc_id in doc_ids]
        
        # Get document texts
        doc_texts = [self.corpus[doc_id] for doc_id in doc_ids]
        
        return {
            'query_id': query_id,
            'query_text': query_text,
            'doc_ids': doc_ids,
            'doc_texts': doc_texts,
            'scores': scores
        }
    

def collate_fn(batch):
    query_ids = [item['query_id'] for item in batch]
    query_texts = [item['query_text'] for item in batch]
    doc_ids = [item['doc_ids'] for item in batch]
    doc_texts = [item['doc_texts'] for item in batch]
    scores = [item['scores'] for item in batch]
    
    # Pad sequences if necessary
    max_docs = max(len(docs) for docs in doc_ids)
    
    padded_doc_ids = [docs + [''] * (max_docs - len(docs)) for docs in doc_ids]
    padded_doc_texts = [texts + [''] * (max_docs - len(texts)) for texts in doc_texts]
    padded_scores = [s + [0] * (max_docs - len(s)) for s in scores]
    
    return {
        'query_ids': query_ids,
        'query_texts': query_texts,
        'doc_ids': padded_doc_ids,
        'doc_texts': padded_doc_texts,
        'scores': torch.tensor(padded_scores)
    }


train_dataset = BEIRDataset(train_queries, train_corpus, train_qrels)
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
    
dev_dataset = BEIRDataset(dev_queries, dev_corpus, dev_qrels)
dev_dataloader = DataLoader(dev_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)

test_dataset = BEIRDataset(test_queries, test_corpus, test_qrels)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)