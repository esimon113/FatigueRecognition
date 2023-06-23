from fatigue_recognition_system import FatigueRecognitionSystem


model_path = "/Users/simon/GitHub/FatigueRecognition/Data/models/other_96_accuracy.h5"


frs = FatigueRecognitionSystem(model_path)
frs.run()

