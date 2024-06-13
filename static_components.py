import streamlit as st
import lgbm_helper
import pandas as pd


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
      st.dataframe(lgbm_helper.get_parameter_df(model), use_container_width=True)
    else:
        st.write('empty')

def model_summary_tabs(model):
  st.write('Model Summary')  
  if model is None :
   model_parameters, model_features , model_trees = st.tabs(["Parameters" , "Features" , "Trees"])
   with model_parameters:
    st.write('empty')
   with  model_features :
    st.write('empty')
   with model_trees :
    st.write('empty')

  else:
      feature_summary_df = lgbm_helper.get_feature_summary_df(model)
      tree_summary_df = lgbm_helper.get_tree_summary(model)
      model_parameters, model_features , model_trees = st.tabs(["Parameters" , f"Features({len(feature_summary_df)})" , f"Trees({len(tree_summary_df)})"])

      with model_parameters:
        #st.write(dir(model_parameters))
        model_summary_parameters_df(model)
      with model_features:
        st.dataframe(feature_summary_df, use_container_width=True)
      with model_trees:
          st.dataframe(tree_summary_df, use_container_width=True)

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
            st.dataframe( pd.DataFrame(df.columns.tolist() , columns=['column']) , use_container_width=True)
        with df_records :
            st.dataframe( df , use_container_width=True)

def dataset_validation(df,model):
    if df is None or model is None :
        st.write('empty')
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

def show_prediction(df,model,valid):
    if df is None or model is None or not valid :
        st.write('empty')
        return False
    else:
        prediction = model.predict(df[lgbm_helper.get_feature_name(model)])
        #st.write(prediction)
        #st.dataframe(prediction)
        st.dataframe( pd.DataFrame(prediction , columns=['prediction']) ,  use_container_width=True)
        return True

def show_prediction_history(df,model,show_prediction):
    
    if df is None or model is None or not show_prediction :
        return
        #st.write('empty')
        #return False
    else:
       bst =  lgbm_helper.get_booster(model)
       tree_detail = bst.trees_to_dataframe()

       tree_index = st.slider("tree index", tree_detail.tree_index.min(), tree_detail.tree_index.max() )
       if len(df) > 1:
         target_index = st.slider("record index" , 0 , len(df) - 1 )
       else:
         target_index = 0
         st.write(f"record index: {target_index}")
