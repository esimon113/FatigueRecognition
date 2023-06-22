import numpy as np
import cv2
import dlib
import time


def calculate_euclidian_distance(p, q):
    x = q[0] - p[0]
    y = q[1] - p[1]

    dist = np.sqrt(np.power(x, 2) + np.power(y, 2))

    return dist



class VisualFatigueDetector:
    def __init__(self, ear_threshold = 0.2, blink_threshold = 5, staring_threshold = 10) -> None:
        self.ear_threshold = ear_threshold
        self.blink_counter = 0
        self.blink_threshold = blink_threshold
        self.fatigue_verdict = False
        self.staring_flag = False
        self.started_staring = 0.0
        self.duration_stared = 0.0
        self.staring_threshold = staring_threshold


    # according to: https://vision.fe.uni-lj.si/cvww2016/proceedings/papers/05.pdf
    def calculate_eye_aspect_ratio(self, eye_landmarks):
        horizontal = calculate_euclidian_distance(eye_landmarks[0], eye_landmarks[3])
        vertical1 = calculate_euclidian_distance(eye_landmarks[1], eye_landmarks[5])
        vertical2 = calculate_euclidian_distance(eye_landmarks[2], eye_landmarks[4])

        ratio = ((vertical1 + vertical2) / 2) / horizontal

        return ratio
    

    def calculate_mouth_aspect_ratio(self, mouth_landmarks):
        X = calculate_euclidian_distance(mouth_landmarks[0], mouth_landmarks[6])
        Y1 = calculate_euclidian_distance(mouth_landmarks[2], mouth_landmarks[10])
        Y2 = calculate_euclidian_distance(mouth_landmarks[4], mouth_landmarks[8])

        ratio = (Y1 + Y2) / 2 / X

        return ratio
        

    def get_eyes(self, landmarks) -> tuple:
        """Return: (left_eye, right_eye)"""
        left_eye_landmarks = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(36, 42)]
        right_eye_landmarks = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(42, 48)]

        return (left_eye_landmarks, right_eye_landmarks)


    def calculate_avg_ear(self, landmarks):
        left_eye, right_eye = self.get_eyes(landmarks)

        left_ear = self.calculate_eye_aspect_ratio(left_eye)
        right_ear = self.calculate_eye_aspect_ratio(right_eye)

        avg_ear = (left_ear + right_ear) / 2

        return avg_ear



    def is_blinking(self, landmarks):
        avg_ear = self.calculate_avg_ear(landmarks)

        if avg_ear < self.ear_threshold:
            self.blink_counter += 1


    def is_staring(self, landmarks):
        ear = self.calculate_avg_ear(landmarks)

        if ear > 0.3:
            if not self.staring_flag:
                self.staring_flag = True
                self.started_staring = time.time()

        else:
            self.staring_flag = False
            self.duration_stared = time.time() - self.started_staring
        
        if self.duration_stared > self.staring_threshold:
            return True





    def is_yawning(self, landmarks):
        pass


    def is_face_dropping(self, landmarks):
        pass


    def is_slow_eye_closing(self, landmarks):
        pass


    def calculate_weighted_visual_fatigue(self, blinking, yawning, staring, faceDropping, slowEyeClosing):
        pass


    # todo: weighted verdict (blink, yawn, ...)
    def calculate_verdict(self):
        if self.blink_counter > self.blink_threshold:
            True



    def run(self):
        face_detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor("../Data/shape_predictor_68_face_landmarks.dat")

        camera = cv2.VideoCapture(0)


        while True:
            ret, frame = camera.read()
            processed_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            detected_faces = face_detector(processed_frame)

            for face in detected_faces:
                # https://pyimagesearch.com/2017/04/03/facial-landmarks-dlib-opencv-python/
                landmarks = predictor(processed_frame, face)


                # TODO: detect rising edge -> count up once per blink...
                if self.is_blinking(landmarks):
                    self.blink_counter += 1
                    print("BLINK")

                    if self.calculate_verdict():
                        print("WAKE UP!!!")
                

                yawning = self.is_yawning(landmarks)
                staring = self.is_staring(landmarks)
                faceDropping = self.is_face_dropping(landmarks)
                slowEyeClosing = self.is_slow_eye_closing(landmarks)

                # hasFatigue = self.calculate_weighted_visual_fatigue(blinking, yawning, staring, faceDropping, slowEyeClosing)


                # mark detected faces and landmarks
                cv2.rectangle(frame, (face.left(), face.top()), (face.right(), face.bottom()), (0, 255, 0), 2)
                # eyes + mouth
                for i in range(36, 67):
                    cv2.circle(frame, (landmarks.part(i).x, landmarks.part(i).y), 1, (0, 0, 255), -1)

                if self.calculate_verdict():
                    print("Fatigue detected...")
                    cv2.putText(frame, "Wake up!!!", (20, 100))

            # Display the frame
            cv2.imshow("Fatigue Detection", frame)
                
            # break on ESC
            key = cv2.waitKey(1)
            if key == 27:
                break
        

        # do clean-up
        camera.release()
        cv2.destroyAllWindows()