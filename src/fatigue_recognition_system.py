import numpy as np
import time
import cv2
from keras.models import load_model
from keras_preprocessing.image import img_to_array



class FatigueRecognitionSystem:
    def __init__(self, model_path) -> None:
        self.face_cascade = cv2.CascadeClassifier("../Data/dataset_yawn_eye/haarcascade/haarcascade_frontalface_default.xml")
        self.left_eye_cascade = cv2.CascadeClassifier("../Data/dataset_yawn_eye/haarcascade/haarcascade_lefteye_2splits.xml")
        self.right_eye_cascade = cv2.CascadeClassifier("../Data/dataset_yawn_eye/haarcascade/haarcascade_righteye_2splits.xml")
        
        self.IMG_SIZE = (145, 145)
        self.has_fatigue = False
        self.initialized_time = time.time()
        self.start_eyes_closed = 0
        self.start_face_dropped = 0

        self.model = load_model(model_path)
        self.camera = cv2.VideoCapture(0)


    def process_eye(self, eye_roi):
        result = -1

        eye = cv2.resize(eye_roi, self.IMG_SIZE)
        eye = eye.astype(float) / 255
        eye = img_to_array(eye)
        eye = np.expand_dims(eye, axis = 0)

        prediction = self.model.predict(eye)
        result = np.argmax(prediction)

        return result
    

    def eye_region_of_interest(self, found_eye_rects, img):
        roi = None

        for (x, y, w, h) in found_eye_rects:
            roi = img[y : y + h, x : x + w]
            break

        return roi


    def process_mouth(self, roi):
        result = -1

        face = cv2.resize(roi, self.IMG_SIZE)
        face = face.astype(float) / 255
        # face = face.reshape((face[0], face[1], face[2], 1))
        face = img_to_array(face)
        face = np.expand_dims(face, axis = 0)

        prediction = self.model.predict(face)
        result = np.argmax(prediction)

        return result





    def is_fatigue(self, parameters: list | tuple, threshold: int):
        yawning = parameters[0] # 4
        left_eye = parameters[1]
        right_eye = parameters[2] #2
        # face_dropped = parameters[3] #2
        # low_voice_energy = parameters[4] #1
        # speak_on_sleep = parameters[5] #1

        fatigue_counter = 0

        print()
        print(f"classes:\n\tyawn: {yawning}\n\tleft: {left_eye}\n\tright: {right_eye}")



        if left_eye == 2 and right_eye == 2:
            if self.start_eyes_closed == 0:
                self.start_eyes_closed = time.time()

            else:
                duration_eyes_closed = time.time() - self.start_eyes_closed

                if duration_eyes_closed > 0.5:
                    weight = np.power(1.1, duration_eyes_closed)

                fatigue_counter += weight

        elif left_eye != 2 or right_eye != 2:
            self.start_eyes_closed = 0

        elif yawning == 3:
            fatigue_counter += 10

        else:
            fatigue_counter = 0
        
        # print(f"fatigue-level: {fatigue_counter}")


        return fatigue_counter > threshold




    def run(self):
        while True:
            _, frame = self.camera.read()

            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            found_face_rects = self.face_cascade.detectMultiScale(gray_frame, 1.1, 5)


            for (x, y, w, h) in found_face_rects:
                gray_face_cropped = gray_frame[y : y + h, x : x + w]

                face_pred = self.process_mouth(gray_face_cropped)


                # Process eyes in the found face
                left_eye_found = self.left_eye_cascade.detectMultiScale(gray_face_cropped)
                right_eye_found = self.right_eye_cascade.detectMultiScale(gray_face_cropped)

                left_eye_roi = self.eye_region_of_interest(left_eye_found, gray_face_cropped)
                right_eye_roi = self.eye_region_of_interest(right_eye_found, gray_face_cropped)

                if left_eye_roi is not None:
                    left_eye_pred = self.process_eye(left_eye_roi)
                else:
                    left_eye_pred = -1
                
                if right_eye_roi is not None:
                    right_eye_pred = self.process_eye(right_eye_roi)
                else:
                    right_eye_pred = -1


                fatigue_parameters: list = []
                fatigue_parameters.append(face_pred)
                fatigue_parameters.append(left_eye_pred)
                fatigue_parameters.append(right_eye_pred)


                if self.is_fatigue(parameters = fatigue_parameters, threshold = 5):
                    cv2.putText(frame, "WAKE UP!!!", (10, 30))
                    print(f"Fatigue detected at {time.time()}")

                    # start chatbot to wake up the user
                


            cv2.imshow("Fatigue Recognition System", frame)
            
            # End on ESC pressed
            key = cv2.waitKey(1)
            if key == 27:
                break
        

        # Clean up
        self.camera.release()
        cv2.destroyAllWindows()

