from dataclasses import dataclass
from autogen_core import SingleThreadedAgentRuntime
from autogen_core import AgentId, MessageContext, RoutedAgent, message_handler
import asyncio
from autogen_core.models import UserMessage
from autogen_ext.models.openai import OpenAIChatCompletionClient

# Create an OpenAI model client.
model_client = OpenAIChatCompletionClient(
    model="gpt-4o-mini",
    api_key="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",
    # api_key="sk-...", # Optional if you have an API key set in the environment.
)

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

    @message_handler
    async def handle_my_message_type(self, message: MyMessageType, ctx: MessageContext) -> None:
        user_message = [UserMessage(
            content=message.content,
            source="user"
        ),]  
        response = await model_client.create(user_message)
        print(response.content)

runtime = SingleThreadedAgentRuntime()

async def main():
    await MyAgent.register(runtime, "Agente1", lambda: MyAgent())
    await MyAgent2.register(runtime, "Agente2", lambda: MyAgent2())
    runtime.start()  # Start processing messages in the background.
    mensaje = input("Ingrese un mensaje: ")
    await runtime.send_message(MyMessageType(mensaje), AgentId("Agente1", "default"))
    await runtime.stop()  # Stop processing messages in the background.

if __name__ == "__main__":
    asyncio.run(main())