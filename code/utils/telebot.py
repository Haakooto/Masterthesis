"""
File for sending messages to telegram. The secret token is stored in a file called secret_teletoken.py
Useful for getting updates on runs on the fly.
"""

import telegram
import utils.secret_teletoken as secret_teletoken

bot_token = secret_teletoken.personal_token
chat_id = secret_teletoken.Bakoton_id

bot = telegram.Bot(bot_token)

def send_img(path, msg):
    with open(path, "rb") as img:
        bot.sendPhoto(chat_id=chat_id, photo=img, caption=msg)

def send_gif(path, msg):
    with open(path, "rb") as img:
        bot.sendDocument(chat_id=chat_id, document=img, caption=msg)

def send_msg(msg):
    bot.sendMessage(chat_id=chat_id, text=msg)

def test():
    bot.sendMessage(chat_id=chat_id, text="testtest")

if __name__ == "__main__":
    test()
