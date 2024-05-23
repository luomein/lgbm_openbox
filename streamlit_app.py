import streamlit as st
import pandas as pd
import numpy as np
import math
import random
import lightgbm as lgb
import pickle
#import joblib
import io
import lgbm_explain
import graphviz

uploaded_model_file = st.file_uploader("Upload your lgbm model file")
upload_record_file =  st.file_uploader("Upload your record", type={"csv"} )

model = None
record = None

if uploaded_model_file is not None:
    model_bytes = uploaded_model_file.read()
    model = pickle.load(io.BytesIO(model_bytes))
    st.write(model.booster_.params['num_iterations'])
if upload_record_file is not None :
    record = pd.read_csv(upload_record_file)
    st.dataframe(data=record)
 

st.write('Hi!')

if (model is not None) and ( record is not None):
  st.write( model.predict(record) )
  st.write( model.predict(record , pred_leaf=True ) )
  st.graphviz_chart( lgb.create_tree_digraph( model, 0 , example_case = record ))
