import speech_recognition as sr
from gtts import gTTS
import pygame
import time

# Initialize the speech recognizer
recognizer = sr.Recognizer()

# Initialize the text-to-speech engine
pygame.mixer.init()

# Function to detect fatigue through speech
def detect_fatigue():
    with sr.Microphone() as source:
        print("Listening for speech...")
        audio = recognizer.listen(source)

        try:
            print("Processing speech...")
            # Use the speech recognizer to convert speech to text
            text = recognizer.recognize_google(audio)
            print("Speech detected:", text)

            # Check for signs of fatigue in the speech
#            if "tired" in text or "sleepy" in text or "exhausted" in text:
            if text in ["tired", "sleepy", "exhausted"]:
                print("Fatigue detected!")

                # Convert the text to speech and play it
                tts = gTTS(text="Please take a break and rest.", lang="en")
                tts.save("fatigue_detected.mp3")
                pygame.mixer.music.load("fatigue_detected.mp3")
                pygame.mixer.music.play()
                time.sleep(5)
                pygame.mixer.music.stop()
        except sr.UnknownValueError:
            print("Speech not recognized.")
        except sr.RequestError as e:
            print("Error occurred:", str(e))

# Call the fatigue detection function
detect_fatigue()
