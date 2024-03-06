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
#import pdb;pdb.set_trace()
model1=model_dt.DT_pred(X,y)
path='saved_models/dt_model.pkl'
# Save the trained model as a pickle file
with open(path, 'wb') as file:
    pickle.dump(model1, file)
model2=model_rf.RF_pred(X,y)
path='saved_models/rf_model.pkl'
# Save the trained model as a pickle file
with open(path, 'wb') as file:
    pickle.dump(model2, file)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=123, test_size=0.20)
model1.fit(X_train, y_train)
model2.fit(X_train, y_train)

plt.figure(figsize=(8, 6))

# Plot ROC curve for Decision Tree model
plot_roc_curve(model1, X_test, y_test, name='Decision Tree')

# Plot ROC curve for Random Forest model
plot_roc_curve(model2, X_test, y_test, name='Random Forest')

# Customize the plot
plt.title('ROC Curves for Multiple Models')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right')

# Show the plot
plt.savefig('ROC.png')