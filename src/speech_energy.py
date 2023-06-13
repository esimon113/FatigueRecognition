import sounddevice as sd
import numpy as np

# Constants for audio recording and analysis
duration = 5  # Recording duration in seconds
fs = 44100  # Sampling frequency
threshold = 0.2  # Energy threshold for fatigue detection

# Function to detect fatigue from speech
def detect_fatigue():
    # Start recording audio
    print("Recording audio...")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()

    # Calculate the energy of the audio signal
    energy = np.sum(audio ** 2)

    # Compare the energy with the threshold for fatigue detection
    if energy < threshold:
        print("Fatigue detected!")
    else:
        print("No fatigue detected.")

# Call the fatigue detection function
detect_fatigue()
