#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
#import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib

# Load the dataset
iris_df = pd.read_csv('Iris.csv')

# Split into features and target variable
X = iris_df.iloc[:, 1:-1].values
y = iris_df.iloc[:, -1].values

# Split into training and testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


# In[2]:


# Standardize the features using a StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Encode the target variable using a LabelEncoder
encoder = LabelEncoder()
encoder.fit(y_train)
y_train_encoded = encoder.transform(y_train)
y_test_encoded = encoder.transform(y_test)


# In[3]:


# Define hyperparameter grid for tuning the model
param_grid = {
    'n_estimators': [25, 50, 100, 200],
    'max_depth': [None, 1, 2, 3],
    'min_samples_split': [2, 3, 4],
    'min_samples_leaf': [1, 2, 3],
    'criterion': ['gini', 'entropy']
}


# In[4]:


# Train a Random Forest Classifier using GridSearchCV to find the optimal hyperparameters
rf_model = RandomForestClassifier(random_state=42)
rf_model_tuned = GridSearchCV(rf_model, param_grid, cv=5)


# In[5]:


rf_model_tuned.fit(X_train_scaled, y_train_encoded)


# In[6]:


# Evaluate the model's performance on the testing and training data
test_score = rf_model_tuned.score(X_test_scaled, y_test_encoded)
train_score = rf_model_tuned.score(X_train_scaled, y_train_encoded)
print(f'Testing score: {test_score:.2f}')
print(f'Training score: {train_score:.2f}')

# Make predictions on the test data using the tuned random forest classifier
y_pred = rf_model_tuned.predict(X_test_scaled)

# Generate a classification report to evaluate the performance of the model
print(f'Classification Report: {classification_report(y_test_encoded, y_pred)}')


# In[7]:


# Save the trained model, StandardScaler, and LabelEncoder for later use
joblib.dump(rf_model_tuned, 'rf_model.sav')
joblib.dump(scaler, 'features_scaler.sav')
joblib.dump(encoder, 'label_encoder.sav')


# In[8]:


import joblib
import streamlit as st


# In[9]:


loaded_model = joblib.load('rf_model.sav')
scaler = joblib.load('features_scaler.sav')
encoder = joblib.load('label_encoder.sav')


# In[10]:


st.write("""
# Iris Flower Prediction App
This app predicts the **Iris flower** type!
""")

col1, col2, col3, col4 = st.columns(4)
sl = col1.slider('Select Sepal Length:', 0.0, 10.0, 5.0)
sw = col2.slider('Select Sepal Width:', 0.0, 10.0, 5.0)
pl = col3.slider('Select Petal Length:', 0.0, 10.0, 5.0)
pw = col4.slider('Select Petal Width:', 0.0, 10.0, 5.0)


# In[11]:


new_data = [[sl, sw, pl, pw]]


# In[12]:


new_data_scaled = scaler.transform(new_data)


# In[13]:


predictions = loaded_model.predict(new_data_scaled)


# In[14]:


decoded_predictions = encoder.inverse_transform(predictions)


# In[15]:


st.write("""
## Prediction
The predicted Iris flower type is:
""")
st.write(decoded_predictions[0])

