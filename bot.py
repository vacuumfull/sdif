import asyncio
import logging
import sys
from os import getenv
from adapter import StableDif
from aiogram import Bot, Dispatcher, Router, types
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from aiogram.filters import CommandStart
from aiogram.types import Message
from aiogram.utils.markdown import hbold

# Bot token can be obtained via https://t.me/BotFather
TOKEN = getenv("BOT_API_KEY")
PROMPT_COMMAND = "/prompt"

IMAGE_URL = './images/'

# All handlers should be attached to the Router (or Dispatcher)
dp = Dispatcher()


@dp.message(CommandStart())
async def command_start_handler(message: Message) -> None:
    """
    This handler receives messages with `/start` command
    """
    # Most event objects have aliases for API methods that can be called in events' context
    # For example if you want to answer to incoming message you can use `message.answer(...)` alias
    # and the target chat will be passed to :ref:`aiogram.methods.send_message.SendMessage`
    # method automatically or call API method directly via
    # Bot instance: `bot.send_message(chat_id=message.chat.id, ...)`
    await message.answer(f"Hello, {hbold(message.from_user.full_name)}!")


@dp.message()
async def echo_handler(message: types.Message) -> None:
    """
    Handler will forward receive a message back to the sender

    By default, message handler will handle all message types (like a text, photo, sticker etc.)
    """
    print(message.text)
    try:
        if PROMPT_COMMAND in message.text:
            text = message.text.replace(PROMPT_COMMAND, '')
            
            sdif = StableDif(use_adapter=False, uid='{message.from_user.id}_{message.message_id}')
            image = sdif.prompt(text)
            filename = sdif.save_image(image)
            print(filename)
            await message.answer_photo(photo=types.FSInputFile(path=f'{IMAGE_URL}{filename}'), caption=text)
            sdif.delete_image(filename)
        # Send a copy of the received message
        await message.send_copy(chat_id=message.chat.id)
    except TypeError as ex:
        # But not all the types is supported to be copied so need to handle it
        print(ex)
        await message.answer("Nice try!")


async def main() -> None:
    # Initialize Bot instance with a default parse mode which will be passed to all API calls
    bot = Bot(TOKEN, default=DefaultBotProperties(parse_mode='HTML'))
    # And the run events dispatching
    await dp.start_polling(bot)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)
    asyncio.run(main())