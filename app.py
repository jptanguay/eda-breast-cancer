'''


Exploratory Data Analysis with Seaborn
https://www.coursera.org/projects/exploratory-data-analysis-seaborn



Dataset: Breast Cancer Wisconsin (Diagnostic)
Source: https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic
credits: Wolberg,William, Mangasarian,Olvi, Street,Nick, and Street,W.. (1995). Breast Cancer Wisconsin (Diagnostic). UCI Machine Learning Repository. https://doi.org/10.24432/C5DW2B.



'''

import streamlit as st
import pandas as pd 


@st.cache_data
def LoadData():
    data = pd.read_csv("data/data.csv")
    return data
    

def datainfo():
    st.title("The complete dataset")
    st.dataframe(data)
    st.write("Description")
    st.dataframe(data.describe())
    
def credits():
    st.title("The complete dataset")
    with open('data/wdbc.names', 'r') as content_file:
        content = content_file.read()    
        st.write(content)

 
##############################
# 
##############################
st.title("EDA - Breast Cancer Wisconsin (Diagnostic)")

data = LoadData()

if 'data' not in st.session_state:
    st.session_state.data = data



     
pg = st.navigation([
    st.Page("home.py", title="Home", icon=":material/home:"),
    st.Page(datainfo, title="Dataset", icon=":material/dataset:"),
    st.Page(credits, title="Credits", icon=":material/group:"),
])
pg.run()

