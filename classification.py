

import streamlit as st

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split, cross_val_score



if 'data' in st.session_state:
    #st.write("Statistics of the DataFrame:")
    #st.write(st.session_state.data.describe())
    data = st.session_state.data
else:
    st.write("No DataFrame found. Please create it in the 'Create DataFrame' page.")


y = data.diagnosis


######################################
# All features
######################################

st.header("Random forest classifier")

st.subheader("All features")
'''
    Classification is performed on all features using sklearn.cross_val_score with 20 folds
'''

drop_cols = ['Unnamed: 32', "id", "diagnosis"]
X = data.drop(drop_cols, axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
rf = RandomForestClassifier()

score = cross_val_score(rf, X, y, cv=20)

c1, c2, c3 = st.columns(3)
with c1:
    st.caption("Features")
    st.write(X.columns)
with c2:
    st.caption("scores")
    st.write(score)
    
with c3:
    st.caption("Result (accuracy)")
    st.metric(label="Mean", value="{:2.3f}".format(score.mean()) )
    st.metric(label="STD", value="{:2.3f}".format(score.std()) )
     

######################################
# Top 5 features
######################################
st.subheader("Selected Features")
'''
    Classification is performed using the top 10 features identified by the mutual information 
'''

X = data[["concave points_mean", "concavity_mean", "area_worst", "smoothness_se", "radius_se", "perimeter_se", "concavity_worst", "area_mean", "concavity_se", "perimeter_mean"]]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
rf = RandomForestClassifier()

score = cross_val_score(rf, X, y, cv=20)

c1, c2, c3 = st.columns(3)
with c1:
    st.caption("Features")
    st.write(X.columns)
with c2:
    st.caption("scores")
    st.write(score)
    
with c3:
    st.caption("Result (accuracy)")
    st.metric(label="Mean", value="{:2.3f}".format(score.mean()) )
    st.metric(label="STD", value="{:2.3f}".format(score.std()) )
      




######################################
# Best reported features
######################################
st.subheader("Best reported features")
'''
    Classification is performed on the three features mentionned in the "Results" paragraph of section 3 (see Credits page)
'''

X = data[["area_worst", "smoothness_worst", "texture_mean"]]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
rf = RandomForestClassifier()

score = cross_val_score(rf, X, y, cv=20)

c1, c2, c3 = st.columns(3)
with c1:
    st.caption("Features")
    st.write(X.columns)
with c2:
    st.caption("scores")
    st.write(score)
    
with c3:
    st.caption("Result (accuracy)")
    st.metric(label="Mean", value="{:2.3f}".format(score.mean()) )
    st.metric(label="STD", value="{:2.3f}".format(score.std()) )
    
    



