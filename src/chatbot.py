from revChatGPT.V3 import Chatbot
from gtts import gTTS
import pygame
import time


# this is just a simple proof of concept



def get_api_key():
    with open(r"../config/chatgpt_api_key.txt") as file:
        key = file.read()

    return key


request = "Tell the world about the ChatGPT API so that a kid can understand it. Keep it short."


# sample for reverse-engineered api:
chatbot = Chatbot(api_key = get_api_key())
result = chatbot.ask(request)
print(result)

# # result = "Hello World!"
file = "../Data/audio/dummySpeach.mp3"

tts = gTTS(text=result, lang="en")
tts.save(file)


pygame.mixer.init()
pygame.mixer.music.load(file)
pygame.mixer.music.play()

# wait until all is played
while pygame.mixer.music.get_busy():
    time.sleep(0.2)

pygame.mixer.music.stop()

