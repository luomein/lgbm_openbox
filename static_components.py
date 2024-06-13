import streamlit as st
import lgbm_helper


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


