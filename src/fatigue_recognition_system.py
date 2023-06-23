import numpy as np
import cv2
from keras.models import load_model
import time



class FatigueRecognitionSystem:
    def __init__(self, model_path) -> None:
        self.face_cascade = cv2.CascadeClassifier("../Data/dataset_yawn_eye/haarcascade/haarcascade_frontalface_default.xml")
        self.left_eye_cascade = cv2.CascadeClassifier("../Data/dataset_yawn_eye/haarcascade/haarcascade_lefteye_2splits.xml")
        self.right_eye_cascade = cv2.CascadeClassifier("../Data/dataset_yawn_eye/haarcascade/haarcascade_righteye_2splits.xml")

        self.camera = cv2.VideoCapture(0)
        self.model = load_model(model_path)


    def run(self):
        while True:
            _, frame = self.camera.read()
            frame_height = frame.shape[0]




