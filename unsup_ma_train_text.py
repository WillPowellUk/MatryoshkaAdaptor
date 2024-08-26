import torch
import torch.nn.functional as F

# Limit PyTorch to use only 8 threads
torch.set_num_threads(8)
torch.set_num_interop_threads(8)

import numpy as np
import os


# Define MatryoshkaAdaptor module - a simple MLP with skip connection
class MatryoshkaAdaptor(torch.nn.Module):
    """
    A PyTorch neural network module that adapts the output of an embedding model
    into a desired output dimension using two linear transformations with a ReLU activation in between.
    Includes a skip connection from input to output.
    """
    def __init__(self, input_output_dim, hidden_dim):
        """
        Initializes the MatryoshkaAdaptor module.
        
        Args:
            input_output_dim: An integer representing the input and output dimension of the module which are equal.
            hidden_dim: An integer representing the hidden dimension of the module.
            
        Returns:
            None
        """
        super(MatryoshkaAdaptor, self).__init__()
        self.input_output_dim = input_output_dim
        self.hidden_dim = hidden_dim
        
        # First linear layer to transform the input dimension to a hidden dimension
        self.linear1 = torch.nn.Linear(input_output_dim, hidden_dim)
        # Second linear layer to transform the hidden dimension to the output dimension which is same as input dimension
        self.linear2 = torch.nn.Linear(hidden_dim, input_output_dim)
        # Activation function to introduce non-linearity
        self.activation = torch.nn.ReLU()

    def forward(self, embedding):
        """
        Forward pass of the MatryoshkaAdaptor module.

        Args:
            embedding: A torch.Tensor of shape (batch_size, input_output_dim) representing the input embeddings.

        Returns:
            output: A torch.Tensor of shape (batch_size, input_output_dim) representing the matryoshka embeddings.
        """
        # Apply the first linear transformation followed by the activation function
        hidden_embedding = self.activation(self.linear1(embedding))
        
        # Apply the second linear transformation to get the final adapted embedding
        adapted_embedding = self.linear2(hidden_embedding)
        
        # Add the skip connection by adding the original embedding to the adapted embedding
        mat_embedding = adapted_embedding + embedding
        mat_embedding = hidden_embedding + embedding


        return mat_embedding
    

import torch
import torch.nn.functional as F

# Equation 1 in paper
def pairwise_similarity_loss(ori_corpus_embeddings, mat_corpus_embeddings, m_dims):
    """
    Computes the pairwise similarity loss between original embeddings and matryoshka embeddings.
    
    Args:
        ori_corpus_embeddings: A tensor of shape (batch_size, embedding_dim) representing the original embeddings.
        mat_corpus_embeddings: A tensor of shape (batch_size, embedding_dim) representing the matryoshka/adapted embeddings.
        m_dims: List of reduced matryoshka dimensionality values.
        
    Returns:
        loss: A scalar tensor representing the mean pairwise similarity loss.
    """

    # Original embeddings only need to be normalized and computed once: 

    # Normalize the embeddings along the embedding dimension to get the cosine similarity
    normalized_ori_corpus_embeddings = F.normalize(ori_corpus_embeddings, p=2, dim=1)
    # Compute the cosine similarity matrices
    original_similarity_matrix = torch.matmul(normalized_ori_corpus_embeddings, normalized_ori_corpus_embeddings.T)

    # Get the indices of the upper triangle of the matrices, excluding the diagonal
    batch_size = ori_corpus_embeddings.size(0)
    i, j = torch.triu_indices(batch_size, batch_size, offset=1)

    # Compute the pairwise cosine similarities
    original_pairwise_similarities = original_similarity_matrix[i, j]

    loss = 0.0
    for m in m_dims:
        # Reduce the matryoshka embeddings to m dimensions
        reduced_mat_corpus_embeddings = mat_corpus_embeddings[:, :m]

        # Normalize the embeddings along the embedding dimension to get the cosine similarity
        normalized_mat_corpus_embeddings = F.normalize(mat_corpus_embeddings, p=2, dim=1)
        
        # Compute the cosine similarity matrices
        matryoshka_similarity_matrix = torch.matmul(normalized_mat_corpus_embeddings, normalized_mat_corpus_embeddings.T)
        
        # Compute the pairwise cosine similarities
        matryoshka_pairwise_similarities = matryoshka_similarity_matrix[i, j]
        
        # Compute the absolute difference between corresponding pairwise similarities
        similarity_differences = torch.abs(original_pairwise_similarities - matryoshka_pairwise_similarities)
        
        # Sum up all the absolute differences to produce the final loss
        loss += torch.sum(similarity_differences)
    
    return loss

# Equation 2 in paper
def topk_similarity_loss(ori_corpus_embeddings, mat_corpus_embeddings, m_dims, k=10):
    """
    Computes the top-k similarity loss between original embeddings and matryoshka embeddings.
    
    Args:
        ori_corpus_embeddings: A tensor of shape (batch_size, embedding_dim) representing the original embeddings.
        mat_corpus_embeddings: A tensor of shape (batch_size, embedding_dim) representing the matryoshka/adapted embeddings.
        m_dims: List of reduced matryoshka dimensionality values.
        k: The number of top similarities to consider (default is 10).
        
    Returns:
        loss: A scalar tensor representing the top-k similarity loss.
    """
    
    # Normalize the original embeddings to get cosine similarity
    normalized_ori_corpus_embeddings = F.normalize(ori_corpus_embeddings, p=2, dim=1)
    
    # Compute the original cosine similarity matrix
    original_similarity_matrix = torch.matmul(normalized_ori_corpus_embeddings, normalized_ori_corpus_embeddings.T)
    
    # Exclude self-similarity by setting the diagonal to a very low value
    original_similarity_matrix.fill_diagonal_(-float('inf'))
    
    # For each embedding, get the top-k similarities for the original embeddings
    original_topk_values, _ = torch.topk(original_similarity_matrix, k, dim=1)
    
    loss = 0.0
    for m in m_dims:
        # Reduce the matryoshka embeddings to m dimensions
        reduced_mat_corpus_embeddings = mat_corpus_embeddings[:, :m]

        # Normalize the reduced matryoshka embeddings to get cosine similarity
        normalized_mat_corpus_embeddings = F.normalize(reduced_mat_corpus_embeddings, p=2, dim=1)
        
        # Compute the cosine similarity matrix for the reduced embeddings
        matryoshka_similarity_matrix = torch.matmul(normalized_mat_corpus_embeddings, normalized_mat_corpus_embeddings.T)
        
        # Exclude self-similarity by setting the diagonal to a very low value
        matryoshka_similarity_matrix.fill_diagonal_(-float('inf'))
        
        # For each embedding, get the top-k similarities for the matryoshka embeddings
        matryoshka_topk_values, _ = torch.topk(matryoshka_similarity_matrix, k, dim=1)
        
        # Compute the absolute difference between the top-k similarities
        similarity_differences = torch.abs(original_topk_values - matryoshka_topk_values)
        
        # Sum up all the absolute differences to accumulate the final loss
        loss += torch.sum(similarity_differences)
    
    return loss


# Equation 3 in paper
def reconstruction_loss(ori_corpus_embeddings, mat_corpus_embeddings, alpha=1.0):
    """
    Computes the reconstruction loss to ensure the matryoshka embeddings do not deviate
    significantly from the original embeddings, and thus act as a regularizer.
    
    Args:
        ori_corpus_embeddings: A tensor of shape (batch_size, embedding_dim) representing the original embeddings.
        mat_corpus_embeddings: A tensor of shape (batch_size, embedding_dim) representing the matryoshka/adapted embeddings.
        alpha: A reconstruction coefficient that controls the weight of the reconstruction term.
        
    Returns:
        loss: A scalar tensor representing the reconstruction loss.
    """
    # Compute the difference between original and matryoshka embeddings
    diff = ori_corpus_embeddings - mat_corpus_embeddings
    
    # Compute the L2 norm of the difference
    loss = torch.norm(diff, p=2, dim=1)
    
    # Return the mean loss over the batch, scaled by alpha
    return alpha * loss.mean()


# Equation 4 in paper
def unsupervised_objective_fn_loss(ori_corpus_embeddings, mat_corpus_embeddings, m_dims,
                                   k=10, alpha=1.0, beta=1.0):
    """
    Computes the overall unsupervised objective function loss as a combination of top-k similarity loss,
    alpha-scaled pairwise similarity loss, and beta-scaled reconstruction loss.
    
    Args:
        ori_corpus_embeddings: A tensor of shape (batch_size, embedding_dim) representing the original embeddings.
        mat_corpus_embeddings: A tensor of shape (batch_size, mat_embedding_dim) representing the matryoshka/adapted embeddings.
        m_dims: List of reduced matryoshka dimensionality values.
        k: The number of top similar embeddings to consider for the top-k similarity loss.
        alpha: A scaling factor for the pairwise similarity loss.
        beta: A scaling factor for the reconstruction loss.
        
    Returns:
        total_loss: A scalar tensor representing the combined unsupervised objective function loss.
    """
    # Compute the individual loss components
    topk_loss = topk_similarity_loss(ori_corpus_embeddings, mat_corpus_embeddings, m_dims, k)
    pairwise_loss = pairwise_similarity_loss(ori_corpus_embeddings, mat_corpus_embeddings, m_dims)
    rec_loss = reconstruction_loss(ori_corpus_embeddings, mat_corpus_embeddings, beta)
    
    # Combine the losses with the given scaling factors
    total_loss = topk_loss + alpha * pairwise_loss + beta * rec_loss
    
    return total_loss


# Equation 5 in paper
def matryoshka_ranking_loss(query_embeddings, corpus_embeddings, relevance_scores, m_dims, k=10):
    """
    Computes the Matryoshka Ranking Loss using optimized matrix operations and normalization.
    
    Args:
        query_embeddings (torch.Tensor): Query embeddings of shape (num_queries, embedding_dim).
        corpus_embeddings (torch.Tensor): Corpus embeddings of shape (num_docs, embedding_dim).
        relevance_scores (torch.Tensor): Relevance scores of shape (num_queries, num_docs).
        m_dims (List[int]): List of reduced dimensionality values.
        k (int): Number of top similar documents to consider for the loss.
    
    Returns:
        torch.Tensor: The computed Matryoshka Ranking Loss.
    """
    
    total_loss = 0.0
    num_queries = query_embeddings.size(0)
    
    for m in m_dims:
        # Reduce embeddings to m dimensions
        reduced_query_embeddings = query_embeddings[:, :m]
        reduced_corpus_embeddings = corpus_embeddings[:, :m]
        
        # Normalize the embeddings to unit vectors
        reduced_query_embeddings = F.normalize(reduced_query_embeddings, p=2, dim=1)
        reduced_corpus_embeddings = F.normalize(reduced_corpus_embeddings, p=2, dim=1)
        
        # Compute cosine similarities
        similarities = torch.matmul(reduced_query_embeddings, reduced_corpus_embeddings.T)
        
        # Get the top k most similar documents for each query
        top_k_similarities, top_k_indices = torch.topk(similarities, k, dim=1, largest=True)
        
        # Gather the corresponding relevance scores
        top_k_relevance = torch.gather(relevance_scores, 1, top_k_indices)
        
        # Calculate pairwise differences for the top-k relevance scores and similarities
        relevance_diff = top_k_relevance.unsqueeze(2) - top_k_relevance.unsqueeze(1)
        sim_diff = top_k_similarities.unsqueeze(2) - top_k_similarities.unsqueeze(1)
        
        # Only consider pairs where the relevance score difference is positive
        positive_diff_mask = (relevance_diff > 0).float()
        
        # Compute the logistic loss with a numerically stable softplus
        log_loss = F.softplus(sim_diff)
        
        # Weight the loss by the relevance difference and accumulate
        weighted_loss = positive_diff_mask * relevance_diff * log_loss
        total_loss += weighted_loss.sum() / num_queries  # Normalize by the number of queries
    
    return total_loss



# Equation 6 in paper
def supervised_objective_fn_loss(ori_query_embeddings, ori_corpus_embeddings, mat_query_embeddings, mat_corpus_embeddings, relevance_scores, m_dims,
                                   k=5, alpha=1.0, beta=1.0, gamma=1.0):
    """
    Computes the overall supervised objective function loss as a combination of top-k similarity loss,
    alpha-scaled pairwise similarity loss, beta-scaled reconstruction loss and gamma-scaled matryoshka ranking loss.
    
    Args:
        ori_corpus_embeddings: A tensor of shape (batch_size, embedding_dim) representing the original embeddings.
        mat_corpus_embeddings: A tensor of shape (batch_size, mat_embedding_dim) representing the matryoshka embeddings.
        relevance_scores: A tensor of shape (batch_size, num_docs) representing the relevance scores.
        m_dims: List of reduced matryoshka dimensionality values.
        k: The number of top similar embeddings to consider for the top-k similarity loss.
        alpha: A scaling factor for the pairwise similarity loss.
        beta: A scaling factor for the reconstruction loss.
        
    Returns:
        total_loss: A scalar tensor representing the combined unsupervised objective function loss.
    """
    # Compute the individual loss components
    #topk_loss = topk_similarity_loss(ori_corpus_embeddings, mat_corpus_embeddings, m_dims, k)
    #pairwise_loss = pairwise_similarity_loss(ori_corpus_embeddings, mat_corpus_embeddings, m_dims)
    #rec_loss = reconstruction_loss(ori_corpus_embeddings, mat_corpus_embeddings, beta)
    ranking_loss = matryoshka_ranking_loss(ori_corpus_embeddings, mat_corpus_embeddings, relevance_scores, m_dims)

    # Combine the losses with the given scaling factors
    #total_loss = topk_loss + alpha * pairwise_loss + beta * rec_loss + gamma * ranking_loss
    total_loss = ranking_loss
    
    return total_loss


import torch
from torch.optim import Adam
import wandb
from tqdm import tqdm

def train(model, mat_adaptor, train_loader, loss_fn, config, run_name):
    """
    Trains the MatryoshkaAdaptor module using the provided training data.

    Args:
        model: A SentenceTransformer model to generate embeddings.
        mat_adaptor: A MatryoshkaAdaptor module to adapt the embeddings.
        train_loader: A DataLoader object for the training dataset.
        loss_fn: A loss function to compute the loss between original and matryoshka embeddings.
        config: A dictionary containing hyperparameters for training.
        run_name: A string representing the name of the save path run.
    Returns:
        None
    """

    # Detect if CUDA is available and set the device accordingly
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Move the model and mat_adaptor to the device
    model.to(device)
    mat_adaptor.to(device)

    # Unpack the hyperparameters
    epochs = config.get('epochs', 5)
    lr = config.get('lr', 1e-3)
    k = config.get('k', 10)  # Top-k similarity loss
    m_dims = config.get('m_dims', [64, 128, 256])  # Matryoshka embedding dimensions
    alpha = config.get('alpha', 1.0)  # Pairwise similarity loss scaling factor (alpha in paper)
    beta = config.get('beta', 1.0)  # Reconstruction loss scaling factor (beta in paper)
    gamma = config.get('gamma', 1.0)  # Ranking loss scaling factor (gamma in paper)

    # Initialize Weights & Biases
    if config.get('wandb', False):
        wandb.init(project="matryoshka-training", config=config)
        config = wandb.config

    # Define an optimizer for the MatryoshkaAdaptor parameters
    optimizer = Adam(mat_adaptor.parameters(), lr=lr)

    # Set embedding model to eval mode (so that gradients only apply to the MatryoshkaAdaptor)
    model.eval()
    
    # Set MatryoshkaAdaptor to training mode
    mat_adaptor.train()

    for epoch in tqdm(range(epochs), desc="Epochs"):
        total_loss = 0

        # Loop over batches in the current epoch
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False):
            if isinstance(batch, list):
                ori_embeddings = model.encode(batch, convert_to_tensor=True).to(device)  # model batched embeddings
                mat_embeddings = mat_adaptor(ori_embeddings)
                loss = loss_fn(ori_embeddings, mat_embeddings, m_dims, k=k, alpha=alpha, beta=beta)

            elif isinstance(batch, dict):
                queries, corpus, relevance_scores = batch['query'], batch['corpus'], batch['relevance'].to(device)
                ori_query_embeddings = model.encode([queries, corpus], convert_to_tensor=True).to(device)
                ori_corpus_embeddings = model.encode(corpus, convert_to_tensor=True).to(device)
                mat_query_embeddings = mat_adaptor(ori_query_embeddings).to(device)
                mat_corpus_embeddings = mat_adaptor(ori_corpus_embeddings).to(device)
                loss = loss_fn(ori_query_embeddings, ori_corpus_embeddings, mat_query_embeddings, mat_corpus_embeddings, relevance_scores, m_dims, k=k, alpha=alpha, beta=beta, gamma=gamma)

            else:
                raise ValueError("Invalid batch format. Please provide a list or dictionary.")

            optimizer.zero_grad()  # Clear previous gradients
            loss.backward()        # Compute gradients
            optimizer.step()       # Update weights

            print(f"Loss: {loss.item():.4f}")
            total_loss += loss.item()

        # Calculate average loss for the epoch
        avg_loss = total_loss / len(train_loader)
        
        
        # Log the average loss to W&B
        if config.get('wandb', False):
            wandb.log({"epoch": epoch + 1, "loss": avg_loss})
        
        # Print average loss for the epoch
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")

        # Save model checkpoint every 1 epochs and on the final epoch
        if (epoch + 1) % 1 == 0 or (epoch + 1) == epochs:
            checkpoint_path = f"{run_name}_epoch_{epoch+1}.pt"
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': mat_adaptor.state_dict(),
                'input_output_dim': mat_adaptor.input_output_dim,
                'hidden_dim': mat_adaptor.hidden_dim,
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, checkpoint_path)
            print(f"Checkpoint saved at {checkpoint_path}")

    # Final save (this is optional if the final epoch is a multiple of 5)
    final_checkpoint_path = f"{run_name}_epoch_{epochs}_final.pt"
    torch.save({
        'epoch': epochs,
        'model_state_dict': mat_adaptor.state_dict(),
        'input_output_dim': mat_adaptor.input_output_dim,
        'hidden_dim': mat_adaptor.hidden_dim,
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': avg_loss,
    }, final_checkpoint_path)
    print(f"Final checkpoint saved at {final_checkpoint_path}")
    

    # Finish the W&B run
    if config.get('wandb', False):
        wandb.finish()


from datasets import load_dataset
from torch.utils.data import DataLoader, random_split
import torch

# Load the dataset
ds_1 = load_dataset("BeIR/nfcorpus", "corpus")
ds_2 = load_dataset("BeIR/webis-touche2020", "corpus")
ds_3 = load_dataset("BeIR/quora", "corpus")
ds_4 = load_dataset("BeIR/scifact", "corpus")

# Access the 'corpus' dataset
dataset_1 = ds_1['corpus']['text']
dataset_2 = ds_2['corpus']['text']
dataset_3 = ds_3['corpus']['text']
dataset_4 = ds_4['corpus']['text']

from sentence_transformers import SentenceTransformer

# Embedding Model
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Matryoshka-Adaptor
input_output_dim = model.get_sentence_embedding_dimension() # Embedding dimension for model (d in paper)
hidden_dim = input_output_dim # Let hidden layer dimension equal the embedding model dimension
mat_adaptor = MatryoshkaAdaptor(input_output_dim, hidden_dim)

hyperparams = {
    'epochs': 2,
    'lr': 5e-3,
    'batch_size': 128,
    'k': 5,  # Top-k similarity loss
    'm_dims': [64, 128, 256],  # Matryoshka embedding dimensions
    'alpha': 1.0,  # Pairwise similarity loss scaling factor (alpha in paper)
    'beta': 1.0,  # reconstruction loss scaling factor (beta in paper)
    'wandb': True
}

import os

for i, dataset in enumerate([dataset_1, dataset_2, dataset_3, dataset_4]):
    # Define the split sizes
    train_size = int(1.0 * len(dataset))
    test_size = len(dataset) - train_size

    if i == 3:
        hyperparams['epochs'] = 5

    # Split the dataset
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    # Create DataLoader for train and test datasets
    train_dataloader = DataLoader(train_dataset, batch_size=hyperparams['batch_size'], shuffle=True)


    # Define the directory path
    run_name = "ckpts/unsupervised_ma/multi-dataset/text-only"

    # Create the directory if it doesn't exist
    os.makedirs(run_name, exist_ok=True)

    train(model, mat_adaptor, train_dataloader, unsupervised_objective_fn_loss, hyperparams, run_name)
