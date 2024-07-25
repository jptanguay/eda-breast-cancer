

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

st.header("Random forest classifier - all features")
'''
    Classification is performed on all features using sklearn.cross_val_score with 20 iterations
'''

drop_cols = ['Unnamed: 32', "id", "diagnosis"]
X = data.drop(drop_cols, axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
rf = RandomForestClassifier()

score = cross_val_score(rf, X, y, cv=20)

c1, c2 = st.columns(2)

with c1:
    st.write(score)
    
with c2:
    st.metric(label="Mean", value="{:2.3f}".format(score.mean()) )
    st.metric(label="STD", value="{:2.3f}".format(score.std()) )
    

######################################
# Selected features
######################################


'''
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)

acc = accuracy_score(y_test, y_pred)
st.write(acc)

'''