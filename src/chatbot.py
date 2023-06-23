from revChatGPT.V3 import Chatbot
from gtts import gTTS
import pygame
import time
import os



class FatigueEvaderChatbot:
    def __init__(self, audio_folder = "..Data/audio") -> None:
        self.api_key = os.environ["OPENAI_API_KEY"]
        self.chatbot = Chatbot(api_key = self.api_key)
        self.audio_folder = audio_folder
        self.conversation = []

        pygame.mixer.init()


    def text_to_speech(self, text):
        filename = str(time.time()) + ".mp3"
        filepath = os.path.join(self.audio_folder, filename)

        tts = gTTS(text = text, lang = "de")
        tts.save(filepath)

        pygame.mixer.music.load(filepath)
        pygame.mixer.music.play()

        while pygame.mixer.music.get_busy():
            time.sleep(0.2)
        
        pygame.mixer.music.stop()



    def run(self, request):
        result = self.chatbot.ask(
            prompt = "".join(self.conversation) + request,
            role = "assistant")
        
        self.conversation.append(request)
        self.conversation.append(result)

        self.text_to_speech(result)

