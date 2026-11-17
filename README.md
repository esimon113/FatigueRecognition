# FatigueRecognition

Course project for *assistance systems*. It explores whether a webcam can notice signs of fatigue and tell the person to wake up.

Cues tried so far include: prolonged closed eyes and blinking, yawning, staring, and a dropping head. Two approaches sit side by side:

- Facial landmarks (dlib) and ratios such as eye aspect ratio, in the smaller scripts under `src/`.
- A Keras CNN trained on a yawn/eye image set, run live in `src/fatigue_recognition_system.py` with Haar face and eye detection. Training is in `src/train_cnn_model*.ipynb`.

Speech energy, keyword spotting, and a spoken chatbot exist only as sketches. Datasets, cascades, the landmark model, and saved weights live in `Data/` and are not included in this repository.
