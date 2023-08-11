import streamlit as st
import tempfile

import cv2

from ultralytics import YOLO

st.title("Face detection app")

# Load the YOLOv8 model
model = YOLO('best.pt')

b = 0
with st.sidebar:
    file = st.file_uploader("Upload video", type=["mp4"])
    n = st.number_input("Select Grid Width", 4, 8)
    col_sid = st.columns(2)
    if col_sid[0].button('Detect faces!'):
        b = 1
    if col_sid[1].button('Stop!'):
        b = 0


stframe = st.empty()

tab1, tab2 = st.tabs(["Video", "Faces"])

with tab1:
    if file:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(file.read())
        vf = cv2.VideoCapture(tfile.name)
        
        st.write('Last detected face')
        example = st.empty()
        with tab2:
            cols = st.columns(n)

            if b:
                idn, x = 0, 0
                while vf.isOpened():
                    # Read a frame from the video
                    success, frame = vf.read()
                    frc = frame.copy()

                    if success:
                        # Run YOLOv8 tracking on the frame, persisting tracks between frames
                        for box in model.track(frame, persist=True)[0].boxes:
                            if box.conf<0.5:
                                continue

                            x1,y1,x2,y2 = map(int, box.xyxy[0])
                            id = int(box.id) if box.id else 0
                            if id>idn:
                                idn = id
                                cols[x%n].image(cv2.resize(frc[y1:y2, x1:x2], (100,100)), channels="BGR", caption=str(x+1))
                                x += 1
                                example.image(cv2.resize(frc[y1:y2, x1:x2], (150,150)), channels="BGR", caption=str(float(box.conf)*100)[:4]+"%")
                            
                            cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),3)
                            cv2.putText(frame,f"id: {id}",(int(x1),int(y1)-15),cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,255,255),2)
                        # Display the annotated frame
                        stframe.image(frame, channels="BGR", width=700)
                    else:
                        break
