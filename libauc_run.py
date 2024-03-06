import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sklearn.metrics as metrics
import pickle
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score,plot_roc_curve
from sklearn.metrics import plot_roc_curve
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from libauc.losses import AUCMLoss
from libauc.optimizers import PESG
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

# Function to train and evaluate the model using k-fold cross-validation
def train_and_evaluate(model, optimizer, criterion, train_loader, val_loader):
    epochs = 100  # Adjust as needed

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

        val_auc = roc_auc_score(val_labels, val_probs)
        print(f'Epoch {epoch + 1}/{epochs}, Validation AUC-ROC: {val_auc}')

    return val_auc,model
df=pd.read_csv('data/feature_label_data_DT.csv')
X=df.drop(['isFraud'], axis=1)
y=df['isFraud']
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=123, test_size=0.20)
input_size=X.shape[1]
output_size = 1  # Binary classification
hidden_size = 207
learning_rate = 0.06721902406122221
total_auc=0
model3 = FFNModel(input_size, hidden_size, output_size)
model3 = model3.cuda()  # Move the model to CUDA
criterion = AUCMLoss()  # Binary Cross Entropy Loss for binary classification
optimizer = optimizer=PESG(model3.parameters(),
                 loss_fn=criterion,
                 lr=learning_rate,
                 momentum=0.9,
                 margin=1.0,
                 epoch_decay=0.003,
                 weight_decay=0.0001)
train_loader = create_dataloader(X_train, y_train, batch_size=560000)
val_loader = create_dataloader(X_test, y_test, batch_size=560000)
val_auc,model3 = train_and_evaluate(model3, optimizer, criterion, train_loader, val_loader)
total_auc += val_auc
print(f'Average test AUC-ROC for LibAUC FFN: {total_auc}')

# plot the roc curve for the model3
model3.eval()
tensor_data = torch.tensor(X_test.values, dtype=torch.float32)
tensor_target = torch.tensor(y_test.values, dtype=torch.float32)
outputs=model3(tensor_data.cuda())
outputs=outputs.cpu().detach().numpy()
fpr, tpr, _ = metrics.roc_curve(y_test.tolist(), outputs)
auc = metrics.auc(fpr, tpr)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label='FFN (AUC = %0.2f)' % auc)
plt.plot([0, 1], [0, 1],'r--')
plt.title('ROC Curves for FFN')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right')
plt.savefig('figs/ROC_FFN_LibAUC.png')
      