import cv2
import dlib

# Load the pre-trained facial landmarks model
predictor = dlib.shape_predictor('path_to_model.dat')

# Load the face detector from dlib
detector = dlib.get_frontal_face_detector()

# Function to detect if head is falling
def is_head_falling(shape):
    # Extract the position of specific facial landmarks
    # You can modify the indices based on the facial landmarks you want to track
    nose_tip = shape.part(30).y
    chin = shape.part(8).y

    # Calculate the vertical distance between the nose tip and chin
    vertical_distance = chin - nose_tip

    # Check if the vertical distance crosses a certain threshold
    if vertical_distance > 20:  # Adjust the threshold as per your requirements
        return True
    else:
        return False

# Initialize dlib's face detector
face_detector = dlib.get_frontal_face_detector()

# Initialize the facial landmarks predictor
facial_landmark_predictor = dlib.shape_predictor('path_to_model.dat')

# Start video capture (assuming webcam)
cap = cv2.VideoCapture(0)

# Main program loop
while True:
    # Read a frame from the video capture
    ret, frame = cap.read()

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame
    faces = face_detector(gray)

    for face in faces:
        # Detect facial landmarks for each face
        landmarks = facial_landmark_predictor(gray, face)

        # Check if head is falling
        if is_head_falling(landmarks):
            print("Head falling detected!")

        # Draw facial landmarks on the frame (optional)
        for point in landmarks.parts():
            cv2.circle(frame, (point.x, point.y), 2, (0, 255, 0), -1)

    # Display the resulting frame
    cv2.imshow("Head Falling Detection", frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close the windows
cap.release()
cv2.destroyAllWindows()


