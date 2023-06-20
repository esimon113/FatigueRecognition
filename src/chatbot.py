from revChatGPT.V3 import Chatbot
from gtts import gTTS
import pygame
import time

# this is just a simple proof of concept



def get_api_key():
    with open(r"../config/chatgpt_api_key.txt") as file:
        key = file.read()

    return key



chatbot = Chatbot(api_key = get_api_key())
# result = chatbot.ask("How are you?")


result = "Hello World!"
file = "../Data/audio/dummySpeach.mp3"

tts = gTTS(text=result, lang="en")
tts.save(file)


pygame.mixer.init()
pygame.mixer.music.load(file)
pygame.mixer.music.play()

time.sleep(5)
pygame.mixer.music.stop()

