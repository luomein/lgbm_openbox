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
import lgbm_explain
import graphviz
import plotly.figure_factory as ff
import plotly.express as px

uploaded_model_file = st.file_uploader("Upload your lgbm model file", type={"txt"})
upload_record_file =  st.file_uploader("Upload your record", type={"csv"} )

model = None
record = None

if uploaded_model_file is not None:
    model_bytes = io.BytesIO( uploaded_model_file.read() )
    #model = pickle.load(io.BytesIO(model_bytes))
    stringio = model_bytes.getvalue()
    stringio = stringio.decode("utf-8")
    #stringio = StringIO(uploaded_model_file.getvalue().decode("utf-8"))
    #stringio = StringIO(uploaded_model_file.getvalue())
    model = lgb.Booster(model_str=stringio)
    st.dataframe(data=model.trees_to_dataframe())
    #st.write(model.booster_.params['num_iterations'])
    #st.write(model.params['num_iterations'])

if upload_record_file is not None :
    record = pd.read_csv(upload_record_file)
    st.dataframe(data=record)
 

#st.write('Hi!')


if (model is not None) and ( record is not None):
  st.write( model.predict(record) )
  st.write( model.predict(record , pred_leaf=True ) )
  
  pred_history, criteria_df, record_df = lgbm_explain.get_pred_history_df(record , model , record['x']>0 )
  st.dataframe(record_df)
  st.dataframe(pred_history)
  st.dataframe(criteria_df)

  tree_index = st.slider("tree index", pred_history.tree_index.min(), pred_history.tree_index.max() )
  st.line_chart(pred_history, x="tree_index", y="accumulated_score" )
  
  fig = px.line(pred_history, x="tree_index", y="accumulated_score" , title='booster prediction')
  st.plotly_chart(fig, theme="streamlit", use_container_width=True)

  #st.dataframe(get_bounded_feature_values_from_model(model , 0 , 3 ) )
  st.graphviz_chart( lgb.create_tree_digraph( model, 0 , example_case = record ))

#### https://github.com/streamlit/streamlit/issues/455#issuecomment-1811044197

### https://docs.streamlit.io/develop/api-reference/charts/st.plotly_chart
### ff.create_distplot
