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

st.sidebar.title("Input")
st.sidebar.markdown("[Model Upload](#model_upload)")
st.sidebar.markdown("[Data Upload](#data_upload)")
st.sidebar.markdown("[Validation](#validation)")
#st.divider()
st.sidebar.title("Analysis")
st.sidebar.markdown("[Prediction](#prediction)")
st.sidebar.markdown("[Booster Detail](#booster)")
st.sidebar.markdown("[Individual Tree Detail](#tree)")




st.header('Model Upload' , anchor = 'model_upload')
static_components.model_txt_hint_expander()
model = lgbm_helper.get_model(st.file_uploader("Upload your lgbm model file", type={"txt"}))
static_components.model_summary_tabs(model)

st.header('Data Upload' , anchor = 'data_upload')
record = None
def get_dataframe(filepath_or_bufferstr):
    if not filepath_or_bufferstr is None:
        return pd.read_csv(filepath_or_bufferstr)
    else :
        return None

df =  get_dataframe( st.file_uploader("Upload your dataset", type={"csv"} ) )
static_components.dataset_summary_tabs(df)


st.header('Validation' , anchor = 'validation')
dataset_validation = static_components.dataset_validation(df,model)

st.header('Prediction', anchor = 'prediction') 
show_prediction = static_components.show_prediction(df,model,dataset_validation)

tree_index, target_index = static_components.show_booster_detail(df,model,show_prediction) 

   
#st.write('Hi!')
def check_df_column_type(df, selected_columns):
  bypass_features = ["pending_error_count_4_8_positive_lag", "pending_error_count_4_8_negative_lag"]
  for c in selected_columns :
    if (not c in df.columns.tolist() )  and (c in bypass_features) :
        df[c] = 0
    if not is_numeric_dtype(df[c]) :
        df.loc[df[c]=='', c] = 0
        df = df.astype({c: float})
        assert is_numeric_dtype(df[c]) , f"{c} is not numeric"
    print(c , is_numeric_dtype(df[c]) , df[c].values)    
  return df


if (model is not None) and ( record is not None) and len(record) > 0 :
  target_index = st.slider("record index" , 0 , len(record) - 1 )
  record = record.iloc[target_index : (target_index + 1) ].reset_index(drop=True)
  assert len(record) == 1 
  record_num = check_df_column_type(record , lgbm_explain.get_feature_name(model) )[ lgbm_explain.get_feature_name(model)]
  
  st.write( model.predict(record_num) )
  st.write( model.predict(record_num , pred_leaf=True ) )
  pred_history, criteria_df, record_df = lgbm_explain.get_pred_history_df(record_num , model ,record_num.index == 0 )
  st.dataframe(record_df)
  st.dataframe(pred_history)
  
  tree_index = st.slider("tree index", pred_history.tree_index.min(), pred_history.tree_index.max() )
   #st.line_chart(pred_history, x="tree_index", y="accumulated_score" )
  st.dataframe(criteria_df[criteria_df.tree_index ==tree_index  ] )


  fig = px.line(pred_history, x="tree_index", y="accumulated_score" , title='booster prediction')
  fig.add_vline(x=tree_index, opacity=0.2 )
  #fig.add_vline(x=2.5, line_width=3, line_dash="dash", line_color="green")
  #fig.add_hrect(y0=0.9, y1=2.6, line_width=0, fillcolor="red", opacity=0.2)
  st.plotly_chart(fig, theme="streamlit", use_container_width=True)



  #st.dataframe(get_bounded_feature_values_from_model(model , 0 , 3 ) )
  st.graphviz_chart( lgb.create_tree_digraph( model, 0 , example_case = record_num ))

#### https://github.com/streamlit/streamlit/issues/455#issuecomment-1811044197

### https://docs.streamlit.io/develop/api-reference/charts/st.plotly_chart
### ff.create_distplot


