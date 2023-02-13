#!/usr/bin/env python
# -*- coding: utf-8 -*-
# This program is dedicated to the public domain under the CC0 license.

"""
Simple Bot to reply to Telegram messages.

First, a few handler functions are defined. Then, those functions are passed to
the Dispatcher and registered at their respective places.
Then, the bot is started and runs until we press Ctrl-C on the command line.

Usage:
Basic Echobot example, repeats messages.
Press Ctrl-C on the command line or send a signal to the process to stop the
bot.
"""

from telegram.ext import Updater, CommandHandler, MessageHandler, Filters
import logging
import sys
from subprocess import Popen, PIPE
sys.path.append('./../')

import utils.secret_teletoken as teletoken

# Enable logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)


def error(update, context):
    """Log Errors caused by Updates."""
    logger.warning('Update "%s" caused error "%s"', update, context.error)

def get_image(update, context):
    """Predict the image sent by the user."""
    # Get the largest version of the image
    file_id = update.message.photo[-1].file_id
    name = update.message.caption if update.message.caption else 'None'

    # if name is not given, make generic name
    fname = f"Predicts/incoming{'_' + name if name else ''}.jpg"

    save = context.bot.get_file(file_id)  # get the image file
    save.download(fname)  # save the image

    cmd = ["python", "telepredictor.py", fname] + ([name] if name else [])
    p = Popen(cmd, stdout=PIPE, stderr=PIPE)
    out, err = p.communicate()
    print(out.decode("utf-8"))
    print(err.decode("utf-8"))
    
    path = out.decode("utf-8").split("\n")[1]

    if path:  # predict the image, and get path to result
        update.message.reply_photo(open(path, "rb"))  # reply with the result
    else:
        update.message.reply_text("No models found, sorry :(")


def main():
    """Start the bot."""
    updater = Updater(teletoken.personal_token, use_context=True)
    ud = updater.dispatcher

    # Register handlers. All images are sent to get_image
    ud.add_handler(MessageHandler(Filters.photo, get_image))
    ud.add_handler(CommandHandler("status", lambda u, c: u.message.reply_text("I'm alive!")))

    # log all errors
    ud.add_error_handler(error)

    # Start the Bot
    updater.start_polling()

    # Run the bot until you press Ctrl-C or the process receives SIGINT,
    # SIGTERM or SIGABRT. This should be used most of the time, since
    # start_polling() is non-blocking and will stop the bot gracefully.
    updater.idle()


if __name__ == '__main__':
    main()
