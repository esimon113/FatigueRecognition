from revChatGPT.V3 import Chatbot
from gtts import gTTS
import pygame
import time
import os


# this is just a simple proof of concept


def get_api_key():
    return os.environ["OPENAI_API_KEY"]


request = "Tell the world about the ChatGPT API so that a kid can understand it. Keep it short."


# sample for reverse-engineered api:
chatbot = Chatbot(api_key = get_api_key())
result = chatbot.ask(request)

print(f"The request: { request }")
print(f"The result: { result }")


audiofile = "../Data/audio/dummySpeach.mp3"
tts = gTTS(text=result, lang="en")
tts.save(audiofile)


pygame.mixer.init()
pygame.mixer.music.load(audiofile)
pygame.mixer.music.play()

# wait until all is played
while pygame.mixer.music.get_busy():
    time.sleep(0.2)

pygame.mixer.music.stop()

