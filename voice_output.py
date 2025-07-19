import pyttsx3

def speak_output(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()
