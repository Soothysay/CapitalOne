import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import KFold
import optuna
import pandas as pd
import numpy as np
import sklearn.metrics as metrics
from libauc.losses import AUCMLoss
from libauc.optimizers import PESG
df=pd.read_csv('data/feature_label_data_DT.csv')

# Define your FFN model
class FFNModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(FFNModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        return x

# Function to create DataLoader from DataFrame
def create_dataloader(data, target, batch_size):
    tensor_data = torch.tensor(data.values, dtype=torch.float32)
    tensor_target = torch.tensor(target.values, dtype=torch.float32)
    dataset = TensorDataset(tensor_data, tensor_target)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader

# Function to define FFN model, optimizer, and loss function
def create_model(optimizer_params, trial):
    # Use trial to sample hyperparameters for tuning
    input_size=df.shape[1]-1
    output_size = 1  # Binary classification
    hidden_size = trial.suggest_int('hidden_size', 64, 512)
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-1)
    
    model = FFNModel(input_size, hidden_size, output_size)
    model = model.cuda()  # Move the model to CUDA
    
    criterion = AUCMLoss()  # Binary Cross Entropy Loss for binary classification
    optimizer=PESG(model.parameters(),
                 loss_fn=criterion,
                 lr=learning_rate,
                 momentum=0.9,
                 margin=1.0,
                 epoch_decay=0.003,
                 weight_decay=0.0001)
    #optimizer = optim.Adam(model.parameters(), lr=learning_rate, **optimizer_params)
    return model, optimizer, criterion

# Function to train and evaluate the model using k-fold cross-validation
def train_and_evaluate(model, optimizer, criterion, train_loader, val_loader):
    epochs = 500  # Adjust as needed

    for epoch in range(epochs):
        model.train()
        for inputs, labels in train_loader:
            inputs, labels = inputs.cuda(), labels.cuda()  # Move data to CUDA
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels.unsqueeze(1))
            loss.backward()
            optimizer.step()

        model.eval()
        val_labels, val_probs = [], []
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.cuda(), labels.cuda()  # Move data to CUDA
                outputs = model(inputs)
                val_labels.extend(labels.cpu().numpy())
                val_probs.extend(outputs.cpu().numpy())

        val_auc = calculate_auc(val_labels, val_probs)
        print(f'Epoch {epoch + 1}/{epochs}, Validation AUC-ROC: {val_auc}')

    return val_auc

# Function to calculate AUC-ROC
def calculate_auc(labels, probs):
    fpr, tpr, _ = metrics.roc_curve(labels, probs)
    auc = metrics.auc(fpr, tpr)
    return auc

# Objective function for Optuna
def objective(trial):
    optimizer_params = {}  # Additional optimizer parameters if needed

    # Initialize a KFold cross-validator
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    total_auc = 0
    for train_idx, val_idx in kf.split(df):
        train_data, val_data = df.iloc[train_idx], df.iloc[val_idx]

        train_loader = create_dataloader(train_data.drop(['isFraud'], axis=1), train_data['isFraud'], batch_size=56000)
        val_loader = create_dataloader(val_data.drop(['isFraud'], axis=1), val_data['isFraud'], batch_size=56000)

        model, optimizer, criterion = create_model(optimizer_params, trial)

        val_auc = train_and_evaluate(model, optimizer, criterion, train_loader, val_loader)
        total_auc += val_auc

    # Return the average validation AUC for hyperparameter optimization
    return total_auc / 5  # Assuming 5-fold cross-validation

# Example usage
if __name__ == "__main__":
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=50)  # Adjust the number of trials as needed

    # Get the best hyperparameters
    best_params = study.best_params
    print("Best Hyperparameters:", best_params)
