import cv2
import dlib

# Load the pre-trained face detector and facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("Data/shape_predictor_68_face_landmarks.dat")

# Initialize variables for yawn detection
yawn_count = 0
yawn_threshold = 20
yawn_flag = False

# Function to calculate the Euclidean distance between two points
def euclidean_dist(pt1, pt2):
    x_diff = pt1[0] - pt2[0]
    y_diff = pt1[1] - pt2[1]
    return (x_diff ** 2 + y_diff ** 2) ** 0.5

# Function to detect yawning based on mouth landmarks
def detect_yawn(shape):
    # Get the coordinates of the mouth landmarks
    mouth_left = shape.part(48)
    mouth_right = shape.part(54)
    mouth_top = shape.part(51)
    mouth_bottom = shape.part(57)

    # Calculate the distance between the top and bottom points of the mouth
    yawn_dist = euclidean_dist((mouth_top.x, mouth_top.y), (mouth_bottom.x, mouth_bottom.y))
    
    return yawn_dist

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

        # Detect yawning
        yawn_dist = detect_yawn(landmarks)
        if yawn_dist > 25:
            if not yawn_flag:
                yawn_flag = True
                yawn_count += 1
                print("Yawn detected!")
        else:
            yawn_flag = False

        # Draw the face bounding box and mouth landmarks on the frame
        cv2.rectangle(frame, (face.left(), face.top()), (face.right(), face.bottom()), (0, 255, 0), 2)
        for i in range(48, 68):
            cv2.circle(frame, (landmarks.part(i).x, landmarks.part(i).y), 1, (0, 0, 255), -1)

    # Display the frame
    cv2.imshow("Fatigue Detection", frame)

    # Check for key press to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close the window
cap.release()
cv2.destroyAllWindows()

# Check if fatigue is detected
if yawn_count >= yawn_threshold:
    print("Fatigue detected!")
else:
    print("No fatigue detected.")
