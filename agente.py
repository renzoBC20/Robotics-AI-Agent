from dataclasses import dataclass
from autogen_core import SingleThreadedAgentRuntime
from autogen_core import AgentId, MessageContext, RoutedAgent, message_handler
import asyncio


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
        await self.send_message(MyMessageType(f"Hola, soy {self.id.type}, recibi este mensaje: {message.content}"), AgentId("Agente2", "default"))

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
