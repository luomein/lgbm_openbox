import streamlit as st
import pandas as pd
import numpy as np
import math
import random
import lightgbm as lgb
import pickle
#import joblib
import io
from io import StringIO
#import lgbm_explain
import graphviz
import plotly.figure_factory as ff
import plotly.express as px
from pandas.api.types import is_numeric_dtype
import static_components
import lgbm_helper

st.sidebar.title("LGBM Openbox")

st.sidebar.header("Input")
st.sidebar.markdown("[Model Upload](#model_upload)")
st.sidebar.markdown("[Data Upload](#data_upload)")
st.sidebar.markdown("[Validation](#validation)")
#st.divider()
st.sidebar.header("Analysis")
st.sidebar.markdown("[Prediction](#prediction)")
st.sidebar.markdown("[Booster Detail](#booster)")
st.sidebar.markdown("[Individual Tree Detail](#tree)")
st.sidebar.divider()



st.header('Model Upload' , anchor = 'model_upload')
static_components.model_txt_hint_expander()
model = lgbm_helper.get_model(st.file_uploader("Upload your lgbm model file", type={"txt"}))
tree_summary_df = static_components.model_summary_tabs(model)
## tree_index = st.slider("tree_index", tree_detail.tree_index.min(), tree_detail.tree_index.max() )
tree_index = None
if tree_summary_df is not None:
  tree_index= st.sidebar.slider("tree_index", tree_summary_df.tree_index.min(), tree_summary_df.tree_index.max() )


st.header('Data Upload' , anchor = 'data_upload')
def get_dataframe(filepath_or_bufferstr):
    if not filepath_or_bufferstr is None:
        return pd.read_csv(filepath_or_bufferstr)
    else :
        return None

df =  get_dataframe( st.file_uploader("Upload your dataset", type={"csv"} ) )
static_components.dataset_summary_tabs(df)

if df is None or len(df) == 0:
  record_index = -1

elif len(df) > 1:
         record_index = st.sidebar.slider("record_index" , 0 , len(df) - 1 )
elif len(df) == 1:
         record_index = 0
         st.sidebar.write(f"record_index: {record_index}")
else :
    record_index = -1

st.header('Validation' , anchor = 'validation')
dataset_validation = static_components.dataset_validation(df,model)

st.header('Prediction', anchor = 'prediction') 
show_prediction = static_components.show_prediction(df,model,dataset_validation , record_index)

static_components.show_booster_detail(df,model,show_prediction,tree_index, record_index) 

   


