import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from models import dt as model_dt
from models import rf as model_rf
from sklearn.metrics import plot_roc_curve
import pickle
from sklearn.model_selection import train_test_split
df=pd.read_csv('data/feature_label_data_DT.csv')
X=df.drop(['isFraud'], axis=1)
y=df['isFraud']
rfm=pickle.load(open('saved_models/rf_model1.pkl', 'rb'))
# Get feature importances
importance = rfm.feature_importances_

# Sort feature importances in descending order
indices = np.argsort(importance)[::-1]

# Rearrange feature names so they match the sorted feature importances
names = [X.columns[i] for i in indices]
# Take top 10 feature importances
top_10_indices = indices[:10]
top_10_names = names[:10]
top_10_importance = importance[top_10_indices]

# Create plot for top 10 feature importances
plt.figure(figsize=(13, 15))
plt.title("Top 10 Feature Importance")
plt.bar(range(10), top_10_importance)
plt.xticks(range(10), top_10_names, rotation=90)
plt.xlabel("Features")
plt.ylabel("Importance")
plt.savefig('figs/feature_importance_RF.png')