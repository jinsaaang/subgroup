import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score

class VIMEEncoder(nn.Module):
    def __init__(self, input_dim, alpha=1.0, hidden_dim=None, device=None):
        """
        VIME Self-supervised encoder with training and transform capability.

        Args:
        - input_dim: dimension of input features
        - alpha: weighting coefficient for feature reconstruction loss
        - hidden_dim: dimension of latent embedding (default = input_dim)
        - device: torch device
        """
        super(VIMEEncoder, self).__init__()
        self.input_dim = input_dim
        self.alpha = alpha
        self.hidden_dim = hidden_dim if hidden_dim is not None else input_dim
        self.device = device or torch.device("cpu")

        self.encoder = nn.Linear(input_dim, self.hidden_dim)
        self.mask_estimator = nn.Linear(self.hidden_dim, input_dim)
        self.feature_estimator = nn.Linear(self.hidden_dim, input_dim)
        self.to(self.device)

    def forward(self, x):
        hidden = torch.relu(self.encoder(x))
        mask_output = torch.sigmoid(self.mask_estimator(hidden))
        feature_output = torch.sigmoid(self.feature_estimator(hidden))
        return mask_output, feature_output

    def train_encoder(self, x_unlab, p_m=0.3, epochs=100, batch_size=64, lr=1e-3, patience=10):
        """
        Train the VIME encoder on unlabeled data using self-supervised objectives.

        Args:
        - x_unlab: torch.Tensor (n_samples, input_dim)
        - p_m: corruption probability
        - epochs: max training epochs
        - batch_size: batch size
        - lr: learning rate
        - patience: early stopping patience
        """
        optimizer = optim.Adam(self.parameters(), lr=lr)
        criterion_mask = nn.BCELoss()
        criterion_feature = nn.MSELoss()

        m_unlab = mask_generator(p_m, x_unlab)
        m_label, x_tilde = pretext_generator(m_unlab, x_unlab)

        x_unlab = x_unlab.float().to(self.device)
        x_tilde = x_tilde.float().to(self.device)
        m_label = m_label.float().to(self.device)

        best_loss = float('inf')
        early_stopping_counter = 0

        for epoch in range(epochs):
            epoch_loss = 0.0
            for i in range(0, len(x_unlab), batch_size):
                inputs = x_tilde[i:i + batch_size]
                labels_mask = m_label[i:i + batch_size]
                labels_feature = x_unlab[i:i + batch_size]

                outputs_mask, outputs_feature = self.forward(inputs)

                loss_mask = criterion_mask(outputs_mask, labels_mask)
                loss_feature = criterion_feature(outputs_feature, labels_feature)
                loss = loss_mask + self.alpha * loss_feature

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            avg_epoch_loss = epoch_loss / len(x_unlab)
            if avg_epoch_loss < best_loss:
                best_loss = avg_epoch_loss
                early_stopping_counter = 0
            else:
                early_stopping_counter += 1
                if early_stopping_counter >= patience:
                    print(f"[VIME] Early stopping at epoch {epoch + 1}")
                    break
            if (epoch + 1) % 10 == 0:
                print(f"[VIME] Epoch [{epoch + 1}/{epochs}] Loss: {avg_epoch_loss:.4f}")

    def encode(self, x):
        """
        Transform input x into latent representation using trained encoder.
        Args:
        - x: torch.Tensor (n_samples, input_dim)
        Returns:
        - z: torch.Tensor (n_samples, latent_dim)
        """
        self.eval()
        with torch.no_grad():
            x = x.float().to(self.device)
            z = torch.relu(self.encoder(x))
        return z
    
def mask_generator(p_m, x):
    """Generate mask tensor in PyTorch.
    
    Args:
    - p_m: corruption probability
    - x: feature tensor
    
    Returns:
    - mask: binary mask tensor 
    """
    # torch.bernoulli is used to generate binary random numbers, 
    # torch.full is used to generate a tensor of the same size as x filled with p_m
    mask = torch.bernoulli(torch.full(x.shape, p_m))
    return mask

def pretext_generator(m, x):  
    """Generate corrupted samples in PyTorch.
  
    Args:
    m: mask tensor
    x: feature tensor
    
    Returns:
    m_new: final mask tensor after corruption
    x_tilde: corrupted feature tensor
    """
    # Parameters
    no, dim = x.shape  
    
    # Randomly (and column-wise) shuffle data
    x_bar = torch.zeros([no, dim])
    for i in range(dim):
        idx = torch.randperm(no)
        x_bar[:, i] = x[idx, i]
    
    # Corrupt samples
    x_tilde = x * (1-m) + x_bar * m  
    # Define new mask tensor
    m_new = 1 * (x != x_tilde)

    return m_new, x_tilde

def perf_metric(metric, y_test, y_test_hat):
    """Evaluate performance in PyTorch.
  
    Args:
    - metric: acc or auc
    - y_test: ground truth label tensor
    - y_test_hat: predicted values tensor
    
    Returns:
    - performance: Accuracy or AUROC performance
    """
    # Convert tensors to numpy arrays for sklearn metrics
    y_test = y_test.numpy()
    y_test_hat = y_test_hat.numpy()

    # Accuracy metric
    if metric == 'acc':
        result = accuracy_score(np.argmax(y_test, axis = 1), 
                                np.argmax(y_test_hat, axis = 1))
    # AUROC metric
    elif metric == 'auc':
        result = roc_auc_score(y_test[:, 1], y_test_hat[:, 1])      
    
    return result

def convert_matrix_to_vector(matrix):
    """Convert two dimensional tensor into one dimensional tensor in PyTorch
  
    Args:
    - matrix: two dimensional tensor
    
    Returns:
    - vector: one dimensional tensor
    """
    # Parameters
    no, dim = matrix.shape
    # Define output  
    vector = torch.zeros([no,], dtype=torch.float)
  
    # Convert matrix to vector
    for i in range(dim):
        idx = (matrix[:, i] == 1).nonzero()
        vector[idx] = i
    
    return vector

def convert_vector_to_matrix(vector):
    """Convert one dimensional tensor into two dimensional tensor in PyTorch
  
    Args:
    - vector: one dimensional tensor
    
    Returns:
    - matrix: two dimensional tensor
    """
    # Parameters
    no = len(vector)
    dim = len(torch.unique(vector))
    # Define output
    matrix = torch.zeros([no,dim])
  
    # Convert vector to matrix
    for i in range(dim):
        idx = (vector == i).nonzero()
        matrix[idx, i] = 1
    
    return matrix