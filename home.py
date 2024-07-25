

import streamlit as st

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns  # data visualization library  
import seaborn.objects as so
import matplotlib.pyplot as plt
import time
from sklearn.feature_selection import mutual_info_classif


if 'data' in st.session_state:
    #st.write("Statistics of the DataFrame:")
    #st.write(st.session_state.data.describe())
    data = st.session_state.data
else:
    st.write("No DataFrame found. Please create it in the 'Create DataFrame' page.")


st.write(data.head())

y = data.diagnosis
drop_cols = ['Unnamed: 32', "id", "diagnosis"]
x = data.drop(drop_cols, axis=1)






"---"
st.header("Features comparison")

'''
    This section compares features in order to identify those who could contain redondant information. Avoiding strongly correlated features in classification tasks
    helps reduce the complexity of the model, thus the risk of errors.

'''



st.subheader("Diagnosis Distributions")

col1, col2 = st.columns(2)
with col1:
    # x=y ?!? that's because we want "y" (diagnosis) on the x axis
    plot = sns.countplot(x=y, label="Count")
    st.pyplot(plot.get_figure())
    
with col2:
    B, M =  y.value_counts()
    st.metric(label="Number of Bening Tumors", value=B)
    st.metric(label="Number of Malignant Tumors", value=M)
    s = "{:2.1f}".format(B/M)
    st.metric(label="Ratio Bening/Malignant", value=s)

'''
The dataset is somewhat unbalanced. Oversampling the M features is to be considered if a classification algorithm is used on the dataset.
'''

#####################
# violin
#####################
"---"
st.subheader("Violin plots of standardized Data")


tab1, tab2, tab3 = st.tabs(["Features 1 to 10 (mean)", "Features 10 to 20 (standard error)", "Features 20 to 30 (worst)"])
tabs = [tab1, tab2, tab3]


data = x
data_std = (data -data.mean()) / data.std()

t = 0
for i in range(0,30,10):
    with tabs[t]:
        data = pd.concat([y, data_std.iloc[:, i:i+10]], axis=1)
        data = pd.melt(data, id_vars="diagnosis",
            var_name="features",
            value_name="value")
            
        plt.figure(figsize=(10,10))
        plot = sns.violinplot(x="features", y="value", hue="diagnosis", data=data, split=True, inner="quart")
        plt.xticks(rotation=45)
        st.pyplot(plot.get_figure())
    t += 1

st.subheader("Correlation between features")

st.write('''

    Columns radius_mean and perimeter_mean seem to be strongly correlated, as well as concavity_mean and concave points_mean.
    
    Other correlated pairs are radius_se and perimeter_se, radius_worst and perimeter_worst, and symmetry_worst and fractal_dimension_worst.
    
    These pairs could hold redondant information.

''')

st.subheader("Separability")
st.write('''

    The features radius_@@@, perimeter_@@@, area_@@@ and concave points_@@@ could be the most separable features. We'll validate that with with swarm plots.

''')




###########################
# joint plots
###########################

"---"
st.subheader("Joint plots for the presumably correlated features ")
'''
    The last two features (symmetry_worst and fractal_dimension_worst) do not seem so correlated all in all.
    
'''
#data = x
col1, col2 = st.columns(2)
for col_val, x_val,y_val in [ (col1, "radius_mean", "perimeter_mean"), (col2, "concavity_mean", "concave points_mean"), (col1, "radius_se", "perimeter_se"), (col2, "symmetry_worst", "fractal_dimension_worst") ]:

    with col_val: 
        jp = sns.jointplot(x=x.loc[:, x_val],
            y=x.loc[:, y_val],
            kind="reg",
            color="#ce1413"
        )
        plt.xticks(rotation=45)

        st.pyplot(jp.figure)


@st.experimental_fragment()
def JointCompare():
    col1, col2, col3 = st.columns( [1,2,1] )
    colnames = x.columns
    x_val = colnames[0]
    y_val = colnames[1]

    with col1:
        y_val = st.selectbox("First column", colnames)

    with col3:
        x_val = st.selectbox("Second column", colnames)
        
    with col2:
        jp = sns.jointplot(x=x.loc[:, x_val],
            y=x.loc[:, y_val],
            kind="reg",
            color="#ce1413"
        )
        plt.xticks(rotation=45)

        st.pyplot(jp.figure)

    
JointCompare()


   

###########################
# Correlation matrix ()
###########################


"---"
st.subheader("Correlation matrix (heatmap)")
'''
    
    
'''
f, ax = plt.subplots(figsize=(18,18))
hm = sns.heatmap(x.corr(), annot=True, linewidth=.5, fmt=".1f", ax=ax, cmap="PiYG")
st.pyplot(hm.get_figure())





###########################
# Features selection
###########################
"---"
st.header("Features selection")

'''
    This section helps visualize which features could offer good separability in a classification tasks.

'''


st.subheader("Swarm plots")

tab1, tab2, tab3 = st.tabs(["Features 1 to 10 (mean)", "Features 10 to 20 (standard error)", "Features 20 to 30 (worst)"])
tabs = [tab1, tab2, tab3]

t = 0
for i in range(0,30,10):
    with tabs[t]: 
        sns.set(style="whitegrid", palette="muted")
        data = x
        data_std = (data -data.mean()) / data.std()
        data = pd.concat([y, data_std.iloc[:, i:i+10]], axis=1)
        data = pd.melt(data, id_vars="diagnosis",
            var_name="features",
            value_name="value")
            
        plt.figure(figsize=(10,10))
        sp = sns.swarmplot(x="features", y="value", hue="diagnosis", data=data)
        plt.xticks(rotation=45)
        st.pyplot(sp.get_figure())
    t += 1
    
    
    
    
###########################
# Mutual information
###########################
"---"
st.subheader("Mutual information")
'''

    The mutual information given by sklearn's feature_selection.mutual_info_classif
    
    Surprisingly, smoothness_se appears in the top positions

'''

@st.cache_data
def make_mi_scores(X, y, discrete_features):
    mi_scores = mutual_info_classif(X, y, discrete_features=discrete_features)
    mi_scores = pd.Series(mi_scores, name="MI Scores", index=X.columns)
    mi_scores = mi_scores.sort_values(ascending=False)
    return mi_scores

discrete_features = x.dtypes == float
mi_scores = make_mi_scores(x, y, discrete_features)
#mi_scores[::3]  # show a few features with their MI scores

col1, col2 = st.columns( [1,2] )

with col1:
    mi_scores

with col2:
    #bp = sns.barplot(mi_scores["MI Scores"])
    plt.figure() #figsize=(10,10))
    bp = sns.barplot(x=mi_scores.values, y=mi_scores.index, orient="y")
    st.pyplot(bp.get_figure())






