import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
import argparse

def predict_video(model, input_video_path, output_video_path, save):
    # Open the input video file
    cap = cv2.VideoCapture(input_video_path)

    # Get video properties
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    #fps = int(cap.get(5))
    fps = 20

    # Define the codec and create a VideoWriter object to save the output video as .MOV
    fourcc = cv2.VideoWriter_fourcc('a', 'v', 'c', '1')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = facecasc.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (255, 0, 0), 2)
            roi_gray = gray[y:y + h, x:x + w]
            cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
            prediction = model.predict(cropped_img)
            maxindex = int(np.argmax(prediction))
            cv2.putText(frame, emotion_dict[maxindex], (x+20, y-60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        # Write the frame to the output video
        if save:
            out.write(frame)

        cv2.imshow('Video', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release video capture and writer objects
    cap.release()
    out.release()
    cv2.destroyAllWindows()

def predict_webcam_and_save(model, output_video_path, save):
    # Open the webcam feed
    cap = cv2.VideoCapture(1)

    # Get webcam properties
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    fps = 20

    # Define the codec and create a VideoWriter object to save the output video
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = facecasc.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (255, 0, 0), 2)
            roi_gray = gray[y:y + h, x:x + w]
            cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
            prediction = model.predict(cropped_img)
            maxindex = int(np.argmax(prediction))
            cv2.putText(frame, emotion_dict[maxindex], (x+20, y-60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        # Write the frame to the output video
        if save:
            out.write(frame)

        cv2.imshow('Video', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release video capture and writer objects
    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Facial Expression Classifier')
    parser.add_argument('--model', required=True, help='Path to the trained model file')
    parser.add_argument('--input_video', help='Path to the user-provided input video file')
    parser.add_argument('--output_video', help='Path to save the output video file')
    args = parser.parse_args()

    model_path = "weights/" + args.model

    if args.input_video:
        input_video_path = "test/" + args.input_video
    else:
        input_video_path = None

    if args.output_video:    
        output_video_path = "result/" + args.output_video
    
    else:
        output_video_path = None

    model = load_model(model_path, compile=False)
    model.compile(optimizer=Adam(learning_rate=1e-3), loss='categorical_crossentropy', metrics=['accuracy'])

    facecasc = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    cv2.ocl.setUseOpenCL(False)
    emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

    if not input_video_path and not output_video_path:
        # Classify using webcam and don't save
        predict_webcam_and_save(model, save=False)

    elif not input_video_path:
        # Classify using webcam and save
        predict_webcam_and_save(model, output_video_path, save=True)

    elif input_video_path and not output_video_path:
        # Classify and save a user-provided video and don't save
        predict_video(model, input_video_path, save=False)

    elif input_video_path and output_video_path:
        # Classify and save a user-provided video and save
        predict_video(model, input_video_path, output_video_path, save=True)