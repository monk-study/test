# %% [markdown]
# # NBA (Next Best Action) Classification Neural Network Implementation
# 
# This notebook implements a neural network approach for the NBA classification problem with the following features:
# - Multi-label classification for 7 NBA classes
# - Special handling for insurance-related NBAs
# - Attention mechanism for feature interactions
# - Class imbalance handling
# - Business rules integration

# %% [markdown]
# ## 1. Import Required Libraries

# %%
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import json
import snowflake.connector
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Check if CUDA is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# %% [markdown]
# ## 2. Data Loading and Preprocessing

# %%
# Snowflake connection parameters
ctx = snowflake.connector.connect(
    authenticator='externalbrowser',
    account='cvs-cvsretailprod.privatelink',
    warehouse='WH_RPHAI_DS_BATCH_01',
    database='CORE_RX',
    schema='CURATED_SCRIPT',
    role='GRP-CN-RPHAI-DS'
)
cs = ctx.cursor()

# Load training data
get_ml_training = """
SELECT *
FROM PL_APP_RPHAI.RAW_RPHAI.TPRR_MODEL_TRAINING_TBL;
"""
cs.execute(get_ml_training)
ml_training_df = cs.fetch_pandas_all()

# %% [markdown]
# ## 3. Feature Processing Functions

# %%
def flatten_value(v):
    # Convert complex values to numeric
    if isinstance(v, list):
        return len(v)
    elif isinstance(v, dict):
        return 1 if v else 0
    elif isinstance(v, (int, float)):
        return float(v)
    elif isinstance(v, str):
        try:
            return float(v)
        except ValueError:
            return 1
    return 0

def process_json_data(data):
    # Process JSON data handling both dict and list formats
    if isinstance(data, list):
        return {
            'list_length': len(data),
            'has_content': 1 if len(data) > 0 else 0
        }
    elif isinstance(data, dict):
        return {k: flatten_value(v) for k, v in data.items()}
    return {}

def process_features(df):
    # Process all features and handle JSON columns
    processed_data = []
    
    for _, row in tqdm(df.iterrows(), total=len(df)):
        feature_dict = {}
        
        # Process numeric columns
        for col in df.select_dtypes(include=[np.number]).columns:
            feature_dict[col] = row[col]
            
        # Process JSON columns
        json_cols = df.select_dtypes(include=['object']).columns
        for col in json_cols:
            try:
                json_data = json.loads(row[col])
                processed = process_json_data(json_data)
                for k, v in processed.items():
                    feature_dict[f"{col}_{k}"] = v
            except:
                feature_dict[f"{col}_missing"] = 1
                
        processed_data.append(feature_dict)
    
    return pd.DataFrame(processed_data)

# %% [markdown]
# ## 4. Dataset Class Implementation

# %%
class NBADataset(Dataset):
    def __init__(self, features, labels, is_training=True):
        # Convert inputs to PyTorch tensors
        self.features = torch.FloatTensor(features.values)
        self.labels = torch.FloatTensor(labels.values)
        self.is_training = is_training
        
        # Calculate positive weights for loss function
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

# %% [markdown]
# ## 5. Neural Network Architecture

# %%
class AttentionBlock(nn.Module):
    def __init__(self, embed_dim, num_heads=8):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads)
        self.norm = nn.LayerNorm(embed_dim)
        
    def forward(self, x):
        # Add sequence dimension for attention
        x_reshaped = x.unsqueeze(0)
        # Apply self-attention
        attn_out, _ = self.attention(x_reshaped, x_reshaped, x_reshaped)
        # Remove sequence dimension
        attn_out = attn_out.squeeze(0)
        # Add residual connection and normalize
        return self.norm(x + attn_out)

class NBANeuralNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dims=[256, 128, 64], num_classes=7):
        super().__init__()
        
        # Input normalization
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
        
        # Attention for feature interactions
        self.attention = AttentionBlock(hidden_dims[-1])
        
        # Insurance-specific branch
        self.insurance_branch = nn.Sequential(
            nn.Linear(hidden_dims[-1], hidden_dims[-1] // 2),
            nn.ReLU(),
            nn.Linear(hidden_dims[-1] // 2, 2)
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
        
        # Apply sigmoid for probability outputs
        return torch.sigmoid(main_preds), torch.sigmoid(insurance_preds)

# %% [markdown]
# ## 6. Loss Function and Training Setup

# %%
class WeightedBCELoss(nn.Module):
    def __init__(self, pos_weights):
        super().__init__()
        self.pos_weights = pos_weights
        
    def forward(self, pred, target):
        # Calculate weighted binary cross-entropy loss
        loss = - (self.pos_weights * target * torch.log(pred + 1e-10) + 
                 (1 - target) * torch.log(1 - pred + 1e-10))
        return loss.mean()

def train_model(model, train_loader, val_loader, device, num_epochs=50):
    # Get positive weights from first batch
    pos_weights = next(iter(train_loader))[0].pos_weights.to(device)
    
    # Initialize loss and optimizer
    criterion = WeightedBCELoss(pos_weights)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3)
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'insurance_f1': [],
        'non_insurance_f1': []
    }
    
    best_val_loss = float('inf')
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0
        for features, labels in tqdm(train_loader):
            features, labels = features.to(device), labels.to(device)
            
            optimizer.zero_grad()
            main_preds, insurance_preds = model(features)
            
            # Calculate losses
            main_loss = criterion(main_preds, labels[:, :5])
            insurance_loss = criterion(insurance_preds, labels[:, 5:])
            total_loss = main_loss + 2 * insurance_loss
            
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_loss += total_loss.item()
        
        # Validation phase
        model.eval()
        val_loss = 0
        val_preds = []
        val_labels = []
        
        with torch.no_grad():
            for features, labels in val_loader:
                features, labels = features.to(device), labels.to(device)
                main_preds, insurance_preds = model(features)
                
                main_loss = criterion(main_preds, labels[:, :5])
                insurance_loss = criterion(insurance_preds, labels[:, 5:])
                total_loss = main_loss + 2 * insurance_loss
                
                val_loss += total_loss.item()
                
                # Store predictions and labels for metrics
                val_preds.append(torch.cat([main_preds, insurance_preds], dim=1).cpu())
                val_labels.append(labels.cpu())
        
        # Calculate metrics
        val_preds = torch.cat(val_preds)
        val_labels = torch.cat(val_labels)
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_nba_model.pt')
        
        # Update history
        history['train_loss'].append(train_loss/len(train_loader))
        history['val_loss'].append(val_loss/len(val_loader))
        
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'Training Loss: {train_loss/len(train_loader):.4f}')
        print(f'Validation Loss: {val_loss/len(val_loader):.4f}')
    
    return history

# %% [markdown]
# ## 7. Prediction and Evaluation Functions

# %%
def predict_nbas(model, features, threshold=0.5):
    # Set model to evaluation mode
    model.eval()
    with torch.no_grad():
        # Get predictions
        main_preds, insurance_preds = model(features)
        
        # Combine predictions
        all_preds = torch.cat([main_preds, insurance_preds], dim=1)
        
        # Get binary predictions and confidence scores
        binary_preds = (all_preds > threshold).float()
        confidence_scores = all_preds
        
        return binary_preds, confidence_scores

def calculate_metrics(y_true, y_pred, threshold=0.5):
    # Convert predictions to binary using threshold
    y_pred_binary = (y_pred > threshold).float()
    
    # Calculate metrics for each class
    metrics = {}
    for i in range(y_true.shape[1]):
        tp = ((y_pred_binary[:, i] == 1) & (y_true[:, i] == 1)).sum()
        fp = ((y_pred_binary[:, i] == 1) & (y_true[:, i] == 0)).sum()
        fn = ((y_pred_binary[:, i] == 0) & (y_true[:, i] == 1)).sum()
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        metrics[f'class_{i}'] = {
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    
    return metrics

# %% [markdown]
# ## 8. Training Pipeline

# %%
def main():
    # 1. Load and preprocess data
    print("Processing features...")
    processed_features = process_features(ml_training_df)
    
    # 2. Split features and labels
    label_columns = ['NBA3_ATTEMPTED', 'NBA4_ATTEMPTED', 'NBA5_ATTEMPTED', 
                    'NBA5_CD_ATTEMPTED', 'NBA7_ATTEMPTED', 'NBA8_ATTEMPTED', 
                    'NBA12_ATTEMPTED']
    
    X = processed_features.drop(label_columns, axis=1)
    y = ml_training_df[label_columns]
    
    # 3. Split data
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y['NBA4_ATTEMPTED']
    )
    
    # 4. Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    # 5. Create data loaders
    train_dataset = NBADataset(X_train_scaled, y_train)
    val_dataset = NBADataset(X_val_scaled, y_val, is_training=False)
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64)
    
    # 6. Initialize and train model
    model = NBANeuralNetwork(input_dim=X_train.shape[1]).to(device)
    history = train_model(model, train_loader, val_loader, device)
    
    # 7. Plot training history
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Training History')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
