import streamlit as st
import lgbm_helper
import pandas as pd
import plotly.figure_factory as ff
import plotly.express as px
import plotly.graph_objects as go
import lightgbm as lgb
import matplotlib.pyplot as plt


def model_txt_hint_expander():
  with st.expander("💡 How to export lgbm model to txt file"):
    st.write('lgbm model')
    code = '''
import lightgbm as lgbm


lgb_model = lgbm.LGBMRegressor()
lgb_model.fit(X_train, y_train, eval_set=[(X_test, y_test)])
lgb_model.booster_.save_model('model.txt')
'''
    st.code(code, language='python')

    st.write('lgbm model booster')
    code2 = '''
import lightgbm as lgbm

train_data = lgbm.Dataset(X_train, label=y_train)
test_data = lgbm.Dataset(X_test, label=y_test)

params = {'objective':'regression'}
lgb_model_booster = lgbm.train(params, train_set=train_data, valid_sets=[test_data])
lgb_model_booster.save_model('model_booster.txt')
'''
    st.code(code2, language='python')

def model_summary_features_df(model):
    if not model is None :
      st.dataframe(lgbm_helper.get_feature_summary_df(model), use_container_width=True)
    else:
        st.write('empty')

def model_summary_parameters_df(model):
    if not model is None :
      #bst =  lgbm_helper.get_booster(model)

      #st.write(bst.params)

      st.dataframe(lgbm_helper.get_parameter_df(model), use_container_width=True)
    else:
        st.write('empty')

def model_summary_tabs(model):
  st.write('Model Summary')  
  if model is None :
   model_parameters, model_features , model_trees , tree_split_gain, feature_split_value = st.tabs(["Parameters" , "Feature Importance" , "Trees", "Tree Split Gain", "Feature Split Value"])
   with model_parameters:
    st.write('empty')
   with  model_features :
    st.write('empty')
   with model_trees :
    st.write('empty')
   with tree_split_gain:
    st.write('empty')
   with feature_split_value:
    st.write('empty')

   return None
  else:
      feature_summary_df = lgbm_helper.get_feature_summary_df(model)
      tree_summary_df = lgbm_helper.get_tree_summary(model)
      bst =  lgbm_helper.get_booster(model)
      tree_detail = bst.trees_to_dataframe()
      model_parameters, model_features , model_trees, tree_split_gain , feature_split_value = st.tabs(["Parameters" , f"Feature Importance({len(feature_summary_df)})" , f"Trees({len(tree_summary_df)})", "Tree Split Gain",  "Feature Split Value"])

      with model_parameters:
        #st.write(dir(model_parameters))
        model_summary_parameters_df(model)
      with model_features:
        st.dataframe(feature_summary_df, use_container_width=True)
      with model_trees:
          st.dataframe(tree_summary_df, use_container_width=True)
      with tree_split_gain:
          fig = px.box(tree_detail, x="tree_index", y="split_gain")
          fig.update_layout(yaxis_type="log")
          st.plotly_chart(fig, use_container_width=True)
      with feature_split_value:
        option = st.selectbox(  "Select Feature", feature_summary_df.Feature.values.tolist() ,  index=None,  placeholder="-----")    
        if not option is None :
          fig, ax = plt.subplots()
          lgb.plot_split_value_histogram(bst , option , ax = ax)
          st.pyplot(fig)
          st.dataframe(tree_detail[tree_detail['split_feature'] == option], use_container_width=True)

      return tree_summary_df
def dataset_summary_tabs(df):
    st.write('Dataset Summary')
    if df is None :
        df_columns, df_records = st.tabs(["Columns" , "Records"])
        with df_columns:
            st.write('empty')
        with df_records :
            st.write('empty')
    else :
        df_columns, df_records = st.tabs([f"Columns({len(df.columns)})" , f"Records({len(df)})"])
        with df_columns:
            describe_df = pd.DataFrame(df.describe()).T.rename_axis('column').reset_index(drop=False)[['column','min','max']]
            df_c = pd.DataFrame(df.columns.tolist() , columns=['column']) 
            df_c = df_c.merge(describe_df , on='column', how='left')
            st.dataframe(df_c , use_container_width=True)
            #st.dataframe( pd.DataFrame(df.columns.tolist() , columns=['column']) , use_container_width=True)
        with df_records :
            st.dataframe( df , use_container_width=True)

def dataset_validation(df,model):
    if df is None or model is None :
        st.write('empty')
        return None
    else:
        if len(df) == 0 :
            st.write('⚠️  Dataset is empty')
            return False
        if set(lgbm_helper.get_feature_name(model)) <= set(df.columns.tolist()):
            st.write('✅ Dataset columns match model features.')
            return True
        else:    
            st.write("⚠️  Dataset columns do not match model features")
            missing_columns = list(set(lgbm_helper.get_feature_name(model)) - set(df.columns.tolist()))
            st.write(f"Missing columns: [{','.join(missing_columns)}]")
    return False

#styled_df = df.style.apply(lambda row: highlight_row(row, row_to_highlight), axis=1)
def highlight_row(row, row_to_highlight_value  ):
    color = 'background-color: yellow'  # Define the highlight color
    #color2 = 'background-color: red' 
    #st.write(row.index)
    #st.write(row.name)
    if row.name == row_to_highlight_value:
        return [color] * len(row)
    else:
        #return [color2] * len(row)

        return [''] * len(row)
def show_prediction(df,model,valid, record_index):
    if df is None or model is None or not valid :
        st.write('empty')
        return False
    else:
        prediction = model.predict(df[lgbm_helper.get_feature_name(model)])
        #st.write(prediction)
        #st.dataframe(prediction)
        prediction_df = pd.DataFrame(prediction , columns=['prediction']).rename_axis('record_index')
        prediction_df = prediction_df.style.apply(lambda row:highlight_row(row, record_index  ) , axis = 1 )
        st.dataframe(prediction_df  ,  use_container_width=True)
        return True

def show_booster_detail(df,model,show_prediction, tree_index,target_index):
    
    if df is None or model is None or not show_prediction :
        st.header('Booster Detail', anchor = 'booster') 
        st.write('empty')
        st.header('Individual Tree Detail', anchor = 'tree') 
        st.write('empty')

        #return None , None
        #return False
    else:
       st.header('Booster Detail', anchor = 'booster') 
       bst =  lgbm_helper.get_booster(model)
       tree_detail = bst.trees_to_dataframe()
       
       record_df = df.iloc[target_index: (target_index + 1)][lgbm_helper.get_feature_name(model)]
       leaf_indices = bst.predict(record_df,  pred_leaf=True)[0]
       prediction = bst.predict( record_df ,  pred_leaf=False)[0]

       leaf_indices = pd.DataFrame(data={'tree_index' : list(range(len(leaf_indices))) , 
                                         'leaf_index' : leaf_indices 
                                         })
       leaf_indices['node_index'] = leaf_indices.apply(lambda row: f"{row['tree_index']}-L{row['leaf_index']}", axis=1)
       leaf_output = tree_detail.merge(leaf_indices , on = ['tree_index' , 'node_index'] , how = 'inner')
       leaf_output['prediction'] = leaf_output['value'].cumsum()
       criteria_df = lgbm_helper.get_booster_nested_criteria(tree_detail ,leaf_indices.tree_index.values.tolist() , leaf_indices.node_index.values.tolist() )
       criteria_df = criteria_df.merge(record_df.T.rename_axis('split_feature').set_axis(['record_value'], axis=1) , on='split_feature'  )
       st.write(f"record_index: {target_index}, prediction: { leaf_output.loc[ leaf_output.tree_index == tree_detail.tree_index.max()  ,'prediction'].values[0] }")
       booster_prediction , split_features = st.tabs(['Booster Prediction', 'Split Features'])
       with booster_prediction:
         
         plot_booster_prediction(leaf_output , tree_index )
       with split_features:
           #st.dataframe(criteria_df)
           show_split_features(df , target_index , criteria_df )
       
       st.header('Individual Tree Detail', anchor = 'tree') 
       st.write(f"tree_index: {tree_index}, leaf_index: { leaf_indices.loc[leaf_indices.tree_index == tree_index , 'node_index'].values[0] }, leaf_output: { leaf_output.loc[leaf_output.tree_index == tree_index , 'value'].values[0] }")
       tree_split_features ,tree_digraph = st.tabs([ 'Tree Split Features' , 'Tree Path'])
       with tree_split_features :
           #st.write('test')
           #st.dataframe(criteria_df[criteria_df.tree_index == tree_index ] ) 
           st.dataframe(criteria_df.loc[criteria_df.tree_index == tree_index
                                        , [ 'tree_index' ,'node_index' , 'split_feature','decision_type','threshold', 'include_na','record_value'] ].set_index('tree_index') 
                        ,  use_container_width=True)

       with tree_digraph:
         st.graphviz_chart( lgb.create_tree_digraph( model, tree_index , example_case = record_df ))


       #return tree_index, target_index
   
def show_split_features(df , record_index , criteria_df ):
    for f in criteria_df.split_feature.unique().tolist():
        with st.expander(f"{f}: {df.iloc[record_index][f]}"):
          st.dataframe(criteria_df.loc[criteria_df.split_feature == f , [ 'tree_index' ,'node_index' , 'split_feature','decision_type','threshold' , 'include_na'] ].set_index('tree_index') 
                        ,  use_container_width=True)
def plot_booster_prediction(leaf_output , tree_index  ):

       fig2 = go.Figure(go.Waterfall(
         name="booster prediction",
         orientation="v",
    measure= ['relative'] * len(leaf_output),
    x=leaf_output['tree_index'],
    #textposition="outside",
    #text=[str(val) for val in values],
    y=leaf_output['value'],
    connector={"line":{"color":"rgb(63, 63, 63)"}},
))
       #if tree_index is not None:
       #if 'tree_index' in st.session_state :
       #    tree_index =st.session_state.tree_index 
       fig2.add_vline(x=tree_index, opacity=0.2 )
       fig2.add_hline(y=leaf_output.loc[leaf_output.tree_index == tree_index , 'prediction' ].values[0] , opacity=0.2 ) 
       
       fig2.update_layout(
    #title="booster prediction",
    xaxis_title="tree_index",
    yaxis_title="prediction",
    showlegend=False
)

       #selection = 
       st.plotly_chart(fig2, theme="streamlit", use_container_width=True
                       #, on_select="rerun"
                       #, selection_mode=('points')
                       )
#{
#  "selection": {
#    "points": [
#      {
#        "curve_number": 0,
#        "point_number": 81,
#        "point_index": 81,
#        "x": 81,
#        "y": 111.09984958888656,
#        "measure": "relative"
#      }
#    ],
#    "point_indices": [
#      81
#    ],
#    "box": [],
#    "lasso": []
#  }
#}
