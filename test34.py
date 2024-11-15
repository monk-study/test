import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.preprocessing import StandardScaler
import json

class NBADataset(Dataset):
    def __init__(self, features, labels, is_training=True):
        """
        Custom dataset for NBA classification
        
        Args:
            features: DataFrame of input features
            labels: DataFrame of NBA labels
            is_training: Whether this is for training (enables data augmentation)
        """
        self.features = torch.FloatTensor(features.values)
        self.labels = torch.FloatTensor(labels.values)
        self.is_training = is_training
        
        # Calculate class weights for weighted loss
        pos_weights = []
        for i in range(labels.shape[1]):
            neg_count = len(labels) - labels.iloc[:, i].sum()
            pos_count = labels.iloc[:, i].sum()
            pos_weights.append(neg_count / pos_count if pos_count > 0 else 1.0)
        self.pos_weights = torch.FloatTensor(pos_weights)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

class AttentionBlock(nn.Module):
    def __init__(self, embed_dim, num_heads=8):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads)
        self.norm = nn.LayerNorm(embed_dim)
        
    def forward(self, x):
        # Reshape input for attention
        x_reshaped = x.unsqueeze(0)  # Add sequence dimension
        attn_out, _ = self.attention(x_reshaped, x_reshaped, x_reshaped)
        attn_out = attn_out.squeeze(0)  # Remove sequence dimension
        return self.norm(x + attn_out)  # Add residual connection

class NBANeuralNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dims=[256, 128, 64], num_classes=7):
        super().__init__()
        
        self.input_norm = nn.BatchNorm1d(input_dim)
        
        # Feature extraction layers
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.3)
            ])
            prev_dim = hidden_dim
        self.feature_layers = nn.Sequential(*layers)
        
        # Attention mechanism for feature interactions
        self.attention = AttentionBlock(hidden_dims[-1])
        
        # Insurance-specific branch
        self.insurance_branch = nn.Sequential(
            nn.Linear(hidden_dims[-1], hidden_dims[-1] // 2),
            nn.ReLU(),
            nn.Linear(hidden_dims[-1] // 2, 2)  # NBA4 and NBA5_CD
        )
        
        # Main classification branch
        self.main_branch = nn.Sequential(
            nn.Linear(hidden_dims[-1], hidden_dims[-1] // 2),
            nn.ReLU(),
            nn.Linear(hidden_dims[-1] // 2, num_classes)
        )
        
    def forward(self, x):
        # Normalize input
        x = self.input_norm(x)
        
        # Extract features
        features = self.feature_layers(x)
        
        # Apply attention
        attended_features = self.attention(features)
        
        # Get predictions from both branches
        insurance_preds = self.insurance_branch(attended_features)
        main_preds = self.main_branch(attended_features)
        
        # Combine predictions
        return torch.sigmoid(main_preds), torch.sigmoid(insurance_preds)

class WeightedBCELoss(nn.Module):
    def __init__(self, pos_weights):
        super().__init__()
        self.pos_weights = pos_weights
        
    def forward(self, pred, target):
        # Calculate weighted BCE loss
        loss = - (self.pos_weights * target * torch.log(pred + 1e-10) + 
                 (1 - target) * torch.log(1 - pred + 1e-10))
        return loss.mean()

def train_model(model, train_loader, val_loader, device, num_epochs=50):
    """Train the neural network"""
    # Get positive weights from first batch
    pos_weights = next(iter(train_loader))[0].pos_weights.to(device)
    
    # Initialize loss and optimizer
    criterion = WeightedBCELoss(pos_weights)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3)
    
    best_val_loss = float('inf')
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0
        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)
            
            optimizer.zero_grad()
            main_preds, insurance_preds = model(features)
            
            # Calculate losses for both branches
            main_loss = criterion(main_preds, labels[:, :5])  # Non-insurance NBAs
            insurance_loss = criterion(insurance_preds, labels[:, 5:])  # Insurance NBAs
            
            # Combined loss with higher weight for insurance cases
            total_loss = main_loss + 2 * insurance_loss
            
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_loss += total_loss.item()
            
        # Validation phase
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for features, labels in val_loader:
                features, labels = features.to(device), labels.to(device)
                main_preds, insurance_preds = model(features)
                
                main_loss = criterion(main_preds, labels[:, :5])
                insurance_loss = criterion(insurance_preds, labels[:, 5:])
                total_loss = main_loss + 2 * insurance_loss
                
                val_loss += total_loss.item()
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_nba_model.pt')
        
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'Training Loss: {train_loss/len(train_loader):.4f}')
        print(f'Validation Loss: {val_loss/len(val_loader):.4f}')

def predict_nbas(model, features, threshold=0.5):
    """Generate NBA predictions with confidence scores"""
    model.eval()
    with torch.no_grad():
        main_preds, insurance_preds = model(features)
        
        # Combine predictions
        all_preds = torch.cat([main_preds, insurance_preds], dim=1)
        
        # Get binary predictions and confidence scores
        binary_preds = (all_preds > threshold).float()
        confidence_scores = all_preds
        
        return binary_preds, confidence_scores
