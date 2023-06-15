import cv2
import dlib
import time

# Load the pre-trained face detector and facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Function to calculate the eye aspect ratio
def eye_aspect_ratio(eye_landmarks):
    A = euclidean_dist(eye_landmarks[1], eye_landmarks[5])
    B = euclidean_dist(eye_landmarks[2], eye_landmarks[4])
    C = euclidean_dist(eye_landmarks[0], eye_landmarks[3])
    return (A + B) / (2.0 * C)

# Function to calculate the Euclidean distance between two points
def euclidean_dist(pt1, pt2):
    x_diff = pt1[0] - pt2[0]
    y_diff = pt1[1] - pt2[1]
    return (x_diff ** 2 + y_diff ** 2) ** 0.5

# Load the webcam
cap = cv2.VideoCapture(0)

# Initialize staring variables
staring_flag = False
staring_start_time = None

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

        # Check for staring
        if avg_eye_aspect_ratio > 0.3:
            if not staring_flag:
                staring_flag = True
                staring_start_time = time.time()
        else:
            if staring_flag:
                staring_flag = False
                staring_duration = time.time() - staring_start_time
                print(f"Staring detected for {staring_duration:.2f} seconds.")

    # Display the frame
    cv2.imshow("Staring Detection", frame)

    # Check for key press to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close the window
cap.release()
cv2.destroyAllWindows()
