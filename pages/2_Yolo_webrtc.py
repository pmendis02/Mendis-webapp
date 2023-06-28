import streamlit as st 
from streamlit_webrtc import webrtc_streamer
import av
from yolo_predictions import YOLO_Pred

#load yolo model 
yolo = YOLO_Pred('./models/best.onnx', './models/data.yaml')




def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")
    
    #flipped = img[::-1,:,:]

    #predicted image 
    pred_image = yolo.predictions(img)

    return av.VideoFrame.from_ndarray(img, format="bgr24")


webrtc_streamer(key="example",
                 video_frame_callback=video_frame_callback, 
                 media_stream_constraints={"video":True,"audio": False })