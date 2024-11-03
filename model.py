import torch
import h5py
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split, Subset
import psutil
import matplotlib.pyplot as plt
import os
from torch.optim.lr_scheduler import ReduceLROnPlateau
import time
import numpy as np
import argparse


class JacobianNet(nn.Module):
    def __init__(self, dropout_rate=0.2):
        super(JacobianNet, self).__init__()
        self.conv1 = nn.Conv2d(2, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.dropout1 = nn.Dropout2d(dropout_rate)
        
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.dropout2 = nn.Dropout2d(dropout_rate)
        
        self.conv3 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(32)
        self.dropout3 = nn.Dropout2d(dropout_rate)
        
        self.conv4 = nn.Conv2d(32, 1, kernel_size=1)

    def forward(self, f, g):
        if f.dim() == 3:
            f = f.unsqueeze(1)
        if g.dim() == 3:
            g = g.unsqueeze(1)
        
        x = torch.cat([f, g], dim=1)
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout2(x)
        
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.dropout3(x)
        
        return self.conv4(x).squeeze(1)
    



class MemoryEfficientDataset(Dataset):

    '''
    1. Load subsets of the dataset in chunks for memory efficiency
    2. Compute statistics about the data
    3. Normalize the data
    '''

    def __init__(self, h5_path, subset_size=None, subset_seed=42, stats=None, chunk_size=1000):
        self.h5_path = h5_path
        self.chunk_size = chunk_size

        # Get dataset length
        with h5py.File(h5_path, 'r') as f:
            total_length = f['f_data'].shape[0]
            
        # Handle subsetting
        if subset_size is not None and subset_size < total_length:
            np.random.seed(subset_seed)
            self.indices = np.random.choice(total_length, subset_size, replace=False)
            self.indices.sort()  # Sort for potentially better HDF5 access patterns
            self.length = subset_size
        else:
            self.indices = None
            self.length = total_length

        if stats is None:
            self.stats = self._compute_statistics()
        else:
            self.stats = stats

    def _compute_statistics(self):
        print("Computing statistics...")
        f_mean = g_mean = j_mean = 0
        f_squared = g_squared = j_squared = 0
        n_chunks = (self.length + self.chunk_size - 1) // self.chunk_size

        with h5py.File(self.h5_path, 'r') as f:
            for chunk_start in range(0, self.length, self.chunk_size):
                chunk_end = min(chunk_start + self.chunk_size, self.length)
                
                if self.indices is not None:
                    chunk_indices = self.indices[chunk_start:chunk_end]
                    f_chunk = torch.tensor(f['f_data'][chunk_indices]).float()
                    g_chunk = torch.tensor(f['g_data'][chunk_indices]).float()
                    j_chunk = torch.tensor(f['jacobian_data'][chunk_indices]).float()
                else:
                    f_chunk = torch.tensor(f['f_data'][chunk_start:chunk_end]).float()
                    g_chunk = torch.tensor(f['g_data'][chunk_start:chunk_end]).float()
                    j_chunk = torch.tensor(f['jacobian_data'][chunk_start:chunk_end]).float()

                f_mean += f_chunk.mean().item() / n_chunks
                g_mean += g_chunk.mean().item() / n_chunks
                j_mean += j_chunk.mean().item() / n_chunks

                f_squared += (f_chunk**2).mean().item() / n_chunks
                g_squared += (g_chunk**2).mean().item() / n_chunks
                j_squared += (j_chunk**2).mean().item() / n_chunks

                del f_chunk, g_chunk, j_chunk

        f_std = (f_squared - f_mean**2)**0.5
        g_std = (g_squared - g_mean**2)**0.5
        j_std = (j_squared - j_mean**2)**0.5

        return {
            'f_mean': f_mean, 'f_std': f_std,
            'g_mean': g_mean, 'g_std': g_std,
            'j_mean': j_mean, 'j_std': j_std
        }

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        with h5py.File(self.h5_path, 'r') as f:
            if self.indices is not None:
                actual_idx = self.indices[idx]
            else:
                actual_idx = idx
                
            f_data = torch.tensor(f['f_data'][actual_idx]).float()
            g_data = torch.tensor(f['g_data'][actual_idx]).float()
            j_data = torch.tensor(f['jacobian_data'][actual_idx]).float()

        # Normalize
        f_norm = (f_data - self.stats['f_mean']) / (self.stats['f_std'] + 1e-8)
        g_norm = (g_data - self.stats['g_mean']) / (self.stats['g_std'] + 1e-8)
        j_norm = (j_data - self.stats['j_mean']) / (self.stats['j_std'] + 1e-8)

        return f_norm, g_norm, j_norm


def save_checkpoint(state, save_dir, is_best=False, is_final=False):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    
    # Save regular checkpoint
    checkpoint_path = os.path.join(save_dir, f'checkpoint_{timestamp}.pt')
    torch.save(state, checkpoint_path)
    print(f"Saved checkpoint to {checkpoint_path}")
    
    if is_best:
        best_path = os.path.join(save_dir, 'best_model.pt')
        torch.save(state, best_path)
        print(f"Saved best model to {best_path}")
    
    if is_final:
        # Save full final model
        final_path = os.path.join(save_dir, 'final_model.pt')
        torch.save(state, final_path)
        
        # Save lightweight inference model
        inference_state = {
            'model_state_dict': state['model_state_dict'],
            'stats': state['stats']
        }
        inference_path = os.path.join(save_dir, 'inference_model.pt')
        torch.save(inference_state, inference_path)
        print(f"Saved final models to {final_path} and {inference_path}")
    
    # Cleanup old checkpoints (keep only last 3)
    checkpoints = [f for f in os.listdir(save_dir) if f.startswith('checkpoint_')]
    checkpoints.sort()
    for old_checkpoint in checkpoints[:-3]:
        os.remove(os.path.join(save_dir, old_checkpoint))

def conservation_loss(pred, true, f, g):

    '''
    The loss function is designed based on the main conservation characteristics of the arakawa scheme
    See paper: 
    {
        Arakawa’s Method Is a Finite-Element Method
            by DENNIS C. JESPERSEN

        PII: 0021-9991(74)90047-3
    }
    '''

    mse = F.mse_loss(pred, true)
    mean_vorticity_loss = torch.abs(pred.mean() - true.mean())
    energy_pred = (f * pred).mean()
    energy_true = (f * true).mean()
    energy_loss = torch.abs(energy_pred - energy_true)
    sq_vorticity_pred = (pred ** 2).mean()
    sq_vorticity_true = (true ** 2).mean()
    sq_vorticity_loss = torch.abs(sq_vorticity_pred - sq_vorticity_true)
    return mse + 0.1 * (mean_vorticity_loss + energy_loss + sq_vorticity_loss)

def train_epoch(model, train_loader, optimizer, device):
    model.train()
    total_loss = 0
    batch_count = 0

    for batch_idx, (f_batch, g_batch, j_true_batch) in enumerate(train_loader):
        f_batch = f_batch.to(device)
        g_batch = g_batch.to(device)
        j_true_batch = j_true_batch.to(device)


        #backward pass
        optimizer.zero_grad()
        j_pred_batch = model(f_batch, g_batch)
        loss = conservation_loss(j_pred_batch, j_true_batch, f_batch, g_batch)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        batch_count += 1

        if batch_idx % 10 == 0:
            memory = torch.cuda.memory_allocated(device) / 1024**2 if torch.cuda.is_available() else psutil.Process().memory_info().rss / 1024**2
            print(f"Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.6f}, Memory: {memory:.1f}MB")
            torch.cuda.empty_cache()

    return total_loss / batch_count

def validate(model, val_loader, device):
    model.eval()
    total_loss = 0
    batch_count = 0

    with torch.no_grad():
        for f_batch, g_batch, j_true_batch in val_loader:
            f_batch = f_batch.to(device)
            g_batch = g_batch.to(device)
            j_true_batch = j_true_batch.to(device)

            j_pred_batch = model(f_batch, g_batch)
            loss = conservation_loss(j_pred_batch, j_true_batch, f_batch, g_batch)

            total_loss += loss.item()
            batch_count += 1

    return total_loss / batch_count

def train_model(data_path, save_dir, subset_size=None, batch_size=32, num_epochs=50, learning_rate=0.001):
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Prepare data
    dataset = MemoryEfficientDataset(data_path, subset_size=subset_size)
    print(f"Using {len(dataset)} samples from the dataset")

    # Split data
    generator = torch.Generator().manual_seed(42)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=generator)

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers= 4, #to be variated accorinding to available computing ressources
        pin_memory=True if torch.cuda.is_available() else False
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=2,
        pin_memory=True if torch.cuda.is_available() else False
    )

    # Initialize model and training components
    model = JacobianNet(dropout_rate=0.2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True, min_lr=1e-6)

    # Training loop setup
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience_counter = 0
    early_stopping_patience = 10

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")

        train_loss = train_epoch(model, train_loader, optimizer, device)
        train_losses.append(train_loss)

        val_loss = validate(model, val_loader, device)
        val_losses.append(val_loss)

        # Learning rate scheduling
        scheduler.step(val_loss)

        # Save checkpoint
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1

        checkpoint_state = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_losses': train_losses,
            'val_losses': val_losses,
            'best_val_loss': best_val_loss,
            'stats': dataset.stats,
            'config': {
                'subset_size': subset_size,
                'batch_size': batch_size,
                'learning_rate': learning_rate
            }
        }

        save_checkpoint(
            state=checkpoint_state,
            save_dir=save_dir,
            is_best=is_best,
            is_final=(epoch == num_epochs-1)
        )

        print(f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
        
        # Plot and save training progress
        plot_losses(train_losses, val_losses, os.path.join(save_dir, 'training_history.png'))

        # Early stopping
        if patience_counter >= early_stopping_patience:
            print(f"\nEarly stopping triggered after {epoch+1} epochs")
            break

        torch.cuda.empty_cache()

    return model, train_losses, val_losses


def load_model_for_inference(model_path, device=None):
    """Load the model and its statistics for inference."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    checkpoint = torch.load(model_path, map_location=device)
    
    model = JacobianNet().to(device)
    
    # Handle different checkpoint formats
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        stats = checkpoint['stats']
    else:
        model.load_state_dict(checkpoint)
        stats = {
            'f_mean': 0.0, 'f_std': 1.0,
            'g_mean': 0.0, 'g_std': 1.0,
            'j_mean': 0.0, 'j_std': 1.0
        }
    
    model.eval()
    return model, stats


def plot_losses(train_losses, val_losses, save_path='training_history.png'):
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Training and Validation Loss Over Time')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(save_path)
    plt.close()



  
if __name__ == "__main__":
    # Create argument parser
    parser = argparse.ArgumentParser(description='Train model with custom data path')
    
    # Add arguments
    parser.add_argument('--data_path', type=str, required=True,
                      help='Path to your training data')
    parser.add_argument('--save_dir', type=str, default="model_checkpoints",
                      help='Directory to save model checkpoints (default: model_checkpoints)')
    parser.add_argument('--subset_size', type=int, default=49000,
                      help='Number of samples to use (default: 49000)')
    parser.add_argument('--batch_size', type=int, default=32,
                      help='Batch size for training (default: 32)')
    parser.add_argument('--num_epochs', type=int, default=50,
                      help='Number of training epochs (default: 50)')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                      help='Learning rate (default: 0.001)')

    # Parse arguments
    args = parser.parse_args()

    # Configuration dictionary from arguments
    CONFIG = {
        'data_path': args.data_path,
        'save_dir': args.save_dir,
        'subset_size': args.subset_size,
        'batch_size': args.batch_size,
        'num_epochs': args.num_epochs,
        'learning_rate': args.learning_rate
    }

    # Create save directory if it doesn't exist
    os.makedirs(CONFIG['save_dir'], exist_ok=True)

    # Train model
    model, train_losses, val_losses = train_model(**CONFIG)