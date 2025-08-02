import asyncio
from ollama import AsyncClient

async def chat():
  message = {'role': 'user', 'content': 'Tell me about fourier decomposition?'}
  async for part in await AsyncClient().chat(model='gemma3', messages=[message], stream=True):
    print(part['message']['content'], end='', flush=True)

asyncio.run(chat())
