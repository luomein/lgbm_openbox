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

def model_summary_configuration_df(model):
    if not model is None :
      st.dataframe(lgbm_helper.get_configuration_df(model), use_container_width=True)
    else:
        st.write('empty')

