import streamlit as st
from yolo_predictions import YOLO_Pred
from PIL import Image
import numpy as np


st.set_page_config(page_title="YOLO Object Detection", layout='wide', page_icon= './images/object.png')

st.title('Welcome to YOLO for Images')

st.write('Please upload Image to get detections')

yolo = YOLO_Pred(onnx_model='./models/best.onnx', data_yaml='./models/data.yaml')

with st.spinner('Please wait while your model is loading'):
    yolo = YOLO_Pred(onnx_model='./models/best.onnx', data_yaml='./models/data.yaml')

    st.balloons()
def upload_image():

    #upload image

    image_file = st.file_uploader(label = 'Upload Image')
    if image_file is not None:
        size_mb = image_file.size/(1024**2)
        file_details = {"filename": image_file.name, "filetype": image_file.type, "filesize": "{:,.2f}MB".format(size_mb)}

        #st.json(file_details)
        #validate file
        if file_details['filetype'] in ('image/png', 'image/jpeg'):
            st.success('Valid image file type') 
            return {"file":image_file, "details":file_details}
        else:
            st.error('Invalid image file type')
            st.error('Upload only PNG or JPG/JPEG')
            return None
def main():
    object = upload_image()

    if object:
        prediction = False
        image_obj = Image.open(object['file'])
        #st.image(image_obj)

        col1, col2 = st.columns(2)

        with col1:
            st.info('Preview of Image')
            st.image(image_obj)

        with col2:
            st.subheader('Check below for file details')
            st.json(object['details'])
            button = st.button('Get Detections from YOLO')
            if button:
                with st.spinner("""
                Getting Objects from image. Please wait
                                """):
                    #below command will convert object to array
                    image_array = np.array(image_obj)
                    pred_img = yolo.predictions(image_array)
                    pred_img_obj = Image.fromarray(pred_img)
                    prediction = True
        if prediction:
            st.subheader("Predicted Image")
            st.caption("Object detection from YOLO V5 model")
            st.image(pred_img_obj)

if __name__ == "__main__":
    main()