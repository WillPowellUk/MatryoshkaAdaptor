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
train_corpus, train_query, train_qrels = GenericDataLoader(data_path).load(split="train")
dev_corpus, dev_query, dev_qrels = GenericDataLoader(data_path).load(split="dev")
test_corpus, test_query, test_qrels = GenericDataLoader(data_path).load(split="test")

class BEIRDataset(Dataset):
    def __init__(self, query, corpus, qrels):
        self.query = query
        self.corpus = corpus
        self.qrels = qrels
        self.query_ids = list(query.keys())
        
    def __len__(self):
        return len(self.query_ids)
    
    def __getitem__(self, idx):
        query_id = self.query_ids[idx]
        query = self.query[query_id]
        
        relevant_docs = self.qrels.get(query_id, {})
        
        # Get all document ids and relevance for this query
        corpus_ids = list(relevant_docs.keys())
        relevance = [relevant_docs[doc_id] for doc_id in corpus_ids]
        
        # Get document texts
        corpus = [self.corpus[doc_id] for doc_id in corpus_ids]
        
        return {
            'query_id': query_id,
            'query': query,
            'corpus_ids': corpus_ids,
            'corpus': corpus,
            'relevance': relevance
        }
    

def collate_fn(batch):
    query_ids = [item['query_id'] for item in batch]
    query = [item['query'] for item in batch]
    corpus_ids = [item['corpus_ids'] for item in batch]
    corpus = [item['corpus'] for item in batch]
    relevance = [item['relevance'] for item in batch]
    
    # Pad sequences if necessary
    max_docs = max(len(docs) for docs in corpus_ids)
    
    padded_corpus_ids = [docs + [''] * (max_docs - len(docs)) for docs in corpus_ids]
    padded_corpus = [texts + [''] * (max_docs - len(texts)) for texts in corpus]
    padded_relevance = [s + [0] * (max_docs - len(s)) for s in relevance]
    
    return {
        'query_ids': query_ids,
        'query': query,
        'corpus_ids': padded_corpus_ids,
        'corpus': padded_corpus,
        'relevance': torch.tensor(padded_relevance)
    }


train_dataset = BEIRDataset(train_query, train_corpus, train_qrels)
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
    
dev_dataset = BEIRDataset(dev_query, dev_corpus, dev_qrels)
dev_dataloader = DataLoader(dev_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)

test_dataset = BEIRDataset(test_query, test_corpus, test_qrels)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)