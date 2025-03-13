import os
from ultralytics import YOLO
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
import time

# Load YOLO model
model_path = "C:/Users/conne/Downloads/Tanay-20241007T171640Z-001/Tanay/mix_50.pt"
model = YOLO(model_path)
output_dir = "extracted_frames"

filepath = 'C:/Users/conne/Downloads/Tanay-20241007T171640Z-001/Tanay/EfficentModel_Mix_20epoch'
direction_model = keras.models.load_model(filepath, compile=False)

def array2dir(array):
    conf = 0.8
    if array[0][0] > conf and array[0][0] > array[0][1] and array[0][0] > array[0][2]:
            print("Left")
            return("Left")
            '''GPIO.output(left_pin, GPIO.HIGH)
            GPIO.output(right_pin, GPIO.LOW)
            GPIO.output(mistake_pin, GPIO.LOW)'''

    elif array[0][1] > conf and array[0][1] > array[0][0] and array[0][1] > array[0][2]:
            print("Right")
            return("Right")
            '''GPIO.output(right_pin, GPIO.HIGH)
            GPIO.output(mistake_pin, GPIO.LOW)
            GPIO.output(left_pin, GPIO.LOW)'''
    else:
            print("No object found")
            return("Mistake!")
            '''GPIO.output(mistake_pin, GPIO.HIGH)
            GPIO.output(left_pin, GPIO.LOW)
            GPIO.output(right_pin, GPIO.LOW)'''
            #send_message("Mistake")


x = [[0.0, 0.0, 0.0]]
x = np.array(x)

# Open the camera
video_path = 'C:/Users/conne/Downloads/Tanay-20241007T171640Z-001/Tanay/test_vid2.mp4'
cap = cv2.VideoCapture(0)

# Set the frame dimensions (adjust based on your camera)
cap.set(3, 640)  # Width
cap.set(4, 480)  # Height

# Set the frame processing rate limit (frames per second)
frame_skip = 5
frame_count = 0

while True:
    # Read a frame from the camera
    ret, frame = cap.read()

    frame_count += 1

    # Process every `frame_interval` frames
    if frame_count % frame_skip == 0:
        # Resize the frame for downsampling
        #frame = cv2.resize(frame, (frame.shape[1] // downsampling_factor, frame.shape[0] // downsampling_factor))

        # Perform object detection
        results = model(frame)[0]

        # Draw bounding boxes and labels
        for result in results.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = result

            if score > 0.5:
                # Extract the frame from the bounding box
                extracted_frame = frame[int(y1):int(y2), int(x1):int(x2)]
                try:
                    img = cv2.resize(extracted_frame, (300, 300))
                except:
                    img = cv2.resize(extracted_frame, (224, 224))
                img = np.asarray(img)
                img = np.expand_dims(img, axis=0)
                output = direction_model.predict(img)
                print(array2dir(output))

                try:
                    ans = array2dir(output)[1]
                except:
                    ans = "Mistake!"
                text = f"{array2dir(output)}, {results.names[int(class_id)].upper()}"
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(frame, text, (int(x1), int(y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)

                # Save the extracted frame
                output_path = os.path.join(output_dir, f"extracted_frame_{ans}_{round(score, 2)}_{class_id}.jpg")
                cv2.imwrite(output_path, extracted_frame)

                print(f"Extracted frame saved at: {output_path}")
    frame_count += 1
    # Display the frame
    cv2.imshow("YOLO Object Detection", frame)

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close the OpenCV windows
cap.release()
cv2.destroyAllWindows()
