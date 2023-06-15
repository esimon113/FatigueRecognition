from visual_fatigue_recognition import VisualFatigueDetector
from audio_fatigue_recognition import AudioFatigueDetector


vfd = VisualFatigueDetector(0.3, 10)
afd = AudioFatigueDetector()



vfd.run()

