#Loan Eligibility prediction 
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sb 
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import LabelEncoder, StandardScaler 
from sklearn import metrics 
from sklearn.svm import SVC 
from imblearn.over_sampling import RandomOverSampler 

import warnings 
warnings.filterwarnings('ignore') 

df = pd.read_csv('loan_data.csv')
df.head()
df.shape
df.info()
df.describe()
temp = df['Loan_Status'].value_counts() 
plt.pie(temp.values, 
		labels=temp.index, 
		autopct='%1.1f%%') 
plt.show() 
plt.subplots(figsize=(15, 5)) 
for i, col in enumerate(['Gender', 'Married']): 
	plt.subplot(1, 2, i+1) 
	sb.countplot(data=df, x=col, hue='Loan_Status') 
plt.tight_layout() 
plt.show() 
plt.subplots(figsize=(15, 5)) 
for i, col in enumerate(['ApplicantIncome', 'LoanAmount']): 
	plt.subplot(1, 2, i+1) 
	sb.distplot(df[col]) 
plt.tight_layout() 
plt.show() 
plt.subplots(figsize=(15, 5)) 
for i, col in enumerate(['ApplicantIncome', 'LoanAmount']): 
	plt.subplot(1, 2, i+1) 
	sb.boxplot(df[col]) 
plt.tight_layout() 
plt.show() 
df = df[df['ApplicantIncome'] < 25000] 
df = df[df['LoanAmount'] < 400000] 
df.groupby('Gender').mean()['LoanAmount']
df.groupby(['Married', 'Gender']).mean()['LoanAmount'] 

def encode_labels(data): 
	for col in data.columns: 
		if data[col].dtype == 'object': 
			le = LabelEncoder() 
			data[col] = le.fit_transform(data[col]) 

	return data 
df = encode_labels(df) 
sb.heatmap(df.corr() > 0.8, annot=True, cbar=False) 
plt.show() 
features = df.drop('Loan_Status', axis=1) 
target = df['Loan_Status'].values 

X_train, X_val,\ 
	Y_train, Y_val = train_test_split(features, target, 
									test_size=0.2, 
									random_state=10) 

ros = RandomOverSampler(sampling_strategy='minority', 
						random_state=0) 
X, Y = ros.fit_resample(X_train, Y_train) 

X_train.shape, X.shape 
scaler = StandardScaler() 
X = scaler.fit_transform(X) 
X_val = scaler.transform(X_val) 
from sklearn.metrics import roc_auc_score 
model = SVC(kernel='rbf') 
model.fit(X, Y) 

print('Training Accuracy : ', metrics.roc_auc_score(Y, model.predict(X))) 
print('Validation Accuracy : ', metrics.roc_auc_score(Y_val, model.predict(X_val))) 
print() 
from sklearn.svm import SVC 
from sklearn.metrics import confusion_matrix 
training_roc_auc = roc_auc_score(Y, model.predict(X)) 
validation_roc_auc = roc_auc_score(Y_val, model.predict(X_val)) 
print('Training ROC AUC Score:', training_roc_auc) 
print('Validation ROC AUC Score:', validation_roc_auc) 
print() 
cm = confusion_matrix(Y_val, model.predict(X_val))
plt.figure(figsize=(6, 6)) 
sb.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False) 
plt.title('Confusion Matrix') 
plt.xlabel('Predicted Label') 
plt.ylabel('True Label') 
plt.show()
from sklearn.metrics import classification_report 
print(classification_report(Y_val, model.predict(X_val))) 




