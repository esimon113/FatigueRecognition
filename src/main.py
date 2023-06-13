import speech_recognition as sr


r = sr.Recognizer()

with sr.Microphone() as src:
    print("Say something!")
    audio = r.listen(src)


try:
    print(f"You (supposedly) said { r.recognize_sphinx(audio) }")
except sr.UnknownValueError:
    print("Audio couldn't be understood")
except sr.RequestError as e:
    print(f"Error: { e }")