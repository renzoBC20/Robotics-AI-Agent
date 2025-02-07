from dataclasses import dataclass
from autogen_core import SingleThreadedAgentRuntime
from autogen_core import AgentId, MessageContext, RoutedAgent, message_handler
import asyncio
from autogen_core.models import UserMessage, AssistantMessage, LLMMessage, SystemMessage
from autogen_ext.models.openai import OpenAIChatCompletionClient
from dotenv import load_dotenv
import os
from typing import List, Any, Dict
# Create an OpenAI model client.
load_dotenv()
openai_api_key = os.getenv('OPENAI_API_KEY')
model_client = OpenAIChatCompletionClient(
    model="gpt-4",
    api_key=openai_api_key,
    # api_key="sk-...", # Optional if you have an API key set in the environment.
)
system_prompt="""
Usted es un robot de exploración que se mueve en cuatro direcciones:
arriba, abajo, derecha e izquierda, como entrada va a recibir una
lectura de temperatura y con ello debes buscar la fuente de calor,
indicando la dirección en que quieres moverte, debes detenerte cuando
creas que estes en la fuente de calor
"""

@dataclass
class MyMessageType:
    content: str

class MyAgent2(RoutedAgent):
    def __init__(self) -> None:
        super().__init__("MyAgent2")

    @message_handler
    async def handle_my_message_type(self, message: MyMessageType, ctx: MessageContext) -> None:
        print(f"Hola, soy {self.id.type}, recibi este mensaje: {message.content}")

class MyAgent(RoutedAgent):
    def __init__(self) -> None:
        super().__init__("MyAgent")
        self.system_message = SystemMessage(content=system_prompt)
        self.chat_history: List[LLMMessage]=[self.system_message]

    @message_handler
    async def handle_my_message_type(self, message: MyMessageType, ctx: MessageContext) -> None:
        user_message = UserMessage(
            content=message.content,
            source="user"
        )
        self.chat_history.append(user_message)
        response = await model_client.create(self.chat_history)
        assistant_message = AssistantMessage(
            content=response.content,
            source="assistant"
        )
        self.chat_history.append(assistant_message)
        print(response.content)

runtime = SingleThreadedAgentRuntime()

async def main():
    await MyAgent.register(runtime, "Agente1", lambda: MyAgent())
    await MyAgent2.register(runtime, "Agente2", lambda: MyAgent2())
    runtime.start()  # Start processing messages in the background.
    
    while True:
        mensaje = input("Ingrese un mensaje (o 'salir' para terminar): ")
        if mensaje.lower() == 'salir':
            break
        await runtime.send_message(MyMessageType(mensaje), AgentId("Agente1", "default"))
    
    await runtime.stop()  # Stop processing messages in the background.

if __name__ == "__main__":
    asyncio.run(main())