import streamlit as st
import json

st.set_page_config(layout="wide")

st.title('Evaluation')

with open('evaluations/evauation.json', 'r') as file:
    evaluation_json = json.load(file)

st.subheader('ROC Curve')

for data_set in ['train','val','test']:
    st.markdown(f'#### {data_set} set')
    col1,col2,col3 = st.columns(3)
    with col1:
        st.image(f'evaluations/confusion_matrix_{data_set}.png')
    with col2:
        st.image(f'evaluations/roc_{data_set}.png')
    with col3:
        st.write(evaluation_json[f'{data_set}'])

st.subheader('Samples')
col1,col2 = st.columns(2)
with col1:
    st.image('evaluations/val_batch0_labels.jpg','val_batch0_labels')
    st.image('evaluations/val_batch1_labels.jpg','val_batch1_labels')
    st.image('evaluations/val_batch2_labels.jpg','val_batch2_labels')

with col2:
    st.image('evaluations/val_batch0_pred.jpg','val_batch0_pred')
    st.image('evaluations/val_batch1_pred.jpg','val_batch1_pred')
    st.image('evaluations/val_batch2_pred.jpg','val_batch2_pred')

st.subheader('Training Process')
st.image('evaluations/results.png')



