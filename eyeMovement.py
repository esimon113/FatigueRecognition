import cv2
import dlib
import numpy as np

# Load the pre-trained face detector and facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Initialize variables for eye detection
eye_aspect_ratio_threshold = 0.3
blink_counter = 0
blink_threshold = 5
fatigue_flag = False

# Function to calculate the Euclidean distance between two points
def euclidean_dist(pt1, pt2):
    x_diff = pt1[0] - pt2[0]
    y_diff = pt1[1] - pt2[1]
    return (x_diff ** 2 + y_diff ** 2) ** 0.5

# Function to calculate the eye aspect ratio
def eye_aspect_ratio(eye_landmarks):
    A = euclidean_dist(eye_landmarks[1], eye_landmarks[5])
    B = euclidean_dist(eye_landmarks[2], eye_landmarks[4])
    C = euclidean_dist(eye_landmarks[0], eye_landmarks[3])
    return (A + B) / (2.0 * C)

# Load the webcam
cap = cv2.VideoCapture(0)

while True:
    # Read a frame from the webcam
    ret, frame = cap.read()

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = detector(gray)

    for face in faces:
        # Detect facial landmarks
        landmarks = predictor(gray, face)

        # Get the eye landmarks
        left_eye_landmarks = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(36, 42)]
        right_eye_landmarks = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(42, 48)]

        # Calculate the eye aspect ratios
        left_eye_aspect_ratio = eye_aspect_ratio(left_eye_landmarks)
        right_eye_aspect_ratio = eye_aspect_ratio(right_eye_landmarks)

        # Calculate the average eye aspect ratio
        avg_eye_aspect_ratio = (left_eye_aspect_ratio + right_eye_aspect_ratio) / 2.0

        # Check for blinking
        if avg_eye_aspect_ratio < eye_aspect_ratio_threshold:
            if not fatigue_flag:
                blink_counter += 1
                print("Blink detected!")
                if blink_counter >= blink_threshold:
                    fatigue_flag = True
                    print("Fatigue detected!")
        else:
            blink_counter = 0
            fatigue_flag = False

        # Draw the face bounding box and eye landmarks on the frame
        cv2.rectangle(frame, (face.left(), face.top()), (face.right(), face.bottom()), (0, 255, 0), 2)
        for i in range(36, 48):
            cv2.circle(frame, (landmarks.part(i).x, landmarks.part(i).y), 1, (0, 0, 255), -1)

    # Display the frame
    cv2.imshow("Fatigue Detection", frame)

    # Check for key press to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close the window
cap.release()
cv2.destroyAllWindows()
