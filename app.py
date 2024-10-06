from PIL import Image
import streamlit as st
from ultralytics import YOLO

model = YOLO('best.pt')

st.title('ระบบตรวจสอบโรคในใบข้าว')
uploaded_files = st.file_uploader("Choose a photo...", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if uploaded_files:
    for uploaded_file in uploaded_files:
        img = Image.open(uploaded_file)
        results = model.predict(img)

        col1,col2 = st.columns(2)
        with col1:
            st.image(img, caption=uploaded_file.name)

        rs = model.predict(img)

        preds_class = []
        preds_confidence = []
        for r in rs:
            for i, t in enumerate(r.probs.top5):
                preds_class.append(r.names[t])
                preds_confidence.append(float(r.probs.top5conf[i]))

        with col2:
            st.write('preds_class', preds_class)
            st.write('preds_confidence', preds_confidence)

        