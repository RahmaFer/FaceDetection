# This is a sample Python script.

# Press Maj+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import cv2
import streamlit as st
face_cascade = cv2.CascadeClassifier('C:/Users/ferja/Desktop/haarcascade_frontalface_default.xml')


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    st.title("Face Detection using Viola-Jones Algorithm")

    color = st.color_picker("Please select a color of the rectangles drawn around the detected faces", "#FF0000")
    rgb_color = tuple(int(color[i:i + 2], 16) for i in (1, 3, 5))
    st.write("Selected RGB color:", rgb_color)
    minNeighbor=st.slider("Select the minNeighbor ",1,10)
    scaleFactor = st.slider("Select the scaleFactor ", 1.1, 2.0, 1.1, 0.1)

    st.subheader("Instructions:")
    st.write("Try to stay in front of the WebCam, please")
    st.write("Press the button below to start detecting faces from your webcam")
    # Add a button to start detecting faces
    if st.button("Detect Faces"):
        # Call the detect_faces function
        # Initialize the webcam
        cap = cv2.VideoCapture(0)

        while True:
            # Read the frames from the webcam
            ret, frame = cap.read()
            # Convert the frames to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # Detect the faces using the face cascade classifier
            faces = face_cascade.detectMultiScale(gray, scaleFactor=scaleFactor, minNeighbors=minNeighbor)
            # Draw rectangles around the detected faces
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), rgb_color, 2)
            # Display the frames
            cv2.imshow('Face Detection using Viola-Jones Algorithm', frame)

            # Exit the loop when 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        # Release the webcam and close all windows
        cv2.imwrite('savedImage.jpg', frame)

        cap.release()
        cv2.destroyAllWindows()









