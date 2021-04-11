# Import Libraries
import cv2
import time

# Load the cascade
smile_cascade = cv2.CascadeClassifier("haarcascade_smile.xml")
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# To capture video from webcam.
video = cv2.VideoCapture(0)

# To use a video file as input
# video = cv2.VideoCapture('filename.mp4')

while True:
    # Read the frame
    check, frame = video.read()
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Detect the faces
    face = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)
    # Draw the rectangle around each face
    for (x, y, w, h) in face:
        img = cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 3)
        # Detect the smile
        smile = smile_cascade.detectMultiScale(
            gray, scaleFactor=1.8, minNeighbors=20)
        # Draw the rectangle around detected smile
        for x, y, w, h in smile:
            img = cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 3)

    # Display output
    cv2.imshow('smile detect', frame)

    # Stop if escape key is pressed
    key = cv2.waitKey(30) & 0xff
    if key == 27:
        break

# Release the VideoCapture object
video.release()
cv2.destroyAllWindows()
