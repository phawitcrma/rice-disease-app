import streamlit as st

st.set_page_config(layout="wide")

st.title('Example Code')

with st.echo():

    # Importing necessary libraries
    from PIL import Image  # For image processing
    import streamlit as st  # For building the web app
    from ultralytics import YOLO  # Importing the YOLO model for object detection

    # Load the YOLO model with the specified weights file
    model = YOLO('best.pt')

    # Specify the path to the image you want to process
    image_path = 'images/IMG_20231018_143921_1.jpg'

    # Open the image using PIL
    im = Image.open(image_path)

    # Use the model to make predictions on the image
    rs = model.predict(im)

    # Initialize lists to store the predicted class names and their corresponding confidence scores
    preds_class = []
    preds_confidence = []

    # Loop through the results returned by the model
    for r in rs:
        # Loop through the top 5 predictions for each detected object
        for i, t in enumerate(r.probs.top5):
            # Append the predicted class name to preds_class
            preds_class.append(r.names[t])
            # Append the confidence score to preds_confidence
            preds_confidence.append(float(r.probs.top5conf[i]))

    # Print the predicted class names and their confidence scores
    print('preds_class', preds_class)
    print('preds_confidence', preds_confidence)

            

col1,col2 = st.columns(2)

with col1:
    st.image(image_path)
    
with col2:
    st.write('preds_class',preds_class)
    st.write('preds_confidence',preds_confidence)



