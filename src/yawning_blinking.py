import cv2
import dlib

# Load the pre-trained face detector and facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("Data/shape_predictor_68_face_landmarks.dat")

# Initialize variables for yawn detection
yawn_count = 0
yawn_threshold = 20
yawn_flag = False

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

# Function to calculate the eye aspect ratio
def eye_aspect_ratio(eye_landmarks):
    A = euclidean_dist(eye_landmarks[1], eye_landmarks[5])
    B = euclidean_dist(eye_landmarks[2], eye_landmarks[4])
    C = euclidean_dist(eye_landmarks[0], eye_landmarks[3])
    return (A + B) / (2.0 * C)




# Load the webcam
camera = cv2.VideoCapture(0)


while True:
    # Read a frame from the webcam
    ret, frame = camera.read()

    # Convert the frame to grayscale
    grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    detectedFaces = detector(grayFrame)

    for face in detectedFaces:
        # Detect facial landmarks
        landmarks = predictor(grayFrame, face)

        # Detect yawning
        yawn_dist = detect_yawn(landmarks)
        if yawn_dist > 25:
            if not yawn_flag:
                yawn_flag = True
                yawn_count += 1
                print("Yawn detected!")
        else:
            yawn_flag = False

        # Get the eye landmarks
        left_eye_landmarks = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(36, 42)]
        right_eye_landmarks = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(42, 48)]

        # Calculate the eye aspect ratios
        left_eye_aspect_ratio = eye_aspect_ratio(left_eye_landmarks)
        right_eye_aspect_ratio = eye_aspect_ratio(right_eye_landmarks)

        # Calculate the average eye aspect ratio
        avg_eye_aspect_ratio = (left_eye_aspect_ratio + right_eye_aspect_ratio) / 2.0


        # Check for blinking and yawning
        if avg_eye_aspect_ratio < eye_aspect_ratio_threshold:
            if not fatigue_flag:
                blink_counter += 1
                print("Blink detected!")
                if blink_counter >= blink_threshold:
                    fatigue_flag = True
                    print("Fatigue detected!")
