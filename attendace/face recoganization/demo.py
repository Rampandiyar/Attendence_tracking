import pyttsx3

# Initialize Text-to-Speech
engine = pyttsx3.init()
engine.setProperty('rate', 150)
engine.setProperty('volume', 0.9)


num = 10
if num > 20:
    print("is too big")
    engine.say(f'20 is too big {num} is small you know')
else:
    print('is too small')
    engine.say(f'{num} is too correct smaller than 20 you know')
engine.runAndWait()