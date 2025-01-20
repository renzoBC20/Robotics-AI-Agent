from dataclasses import dataclass
from autogen_core import SingleThreadedAgentRuntime
from autogen_core import AgentId, MessageContext, RoutedAgent, message_handler
import asyncio


@dataclass
class MyMessageType:
    content: str


class MyAgent(RoutedAgent):
    def __init__(self) -> None:
        super().__init__("MyAgent")

    @message_handler
    async def handle_my_message_type(self, message: MyMessageType, ctx: MessageContext) -> None:
        print(f"{self.id.type} received message: {message.content}")

runtime = SingleThreadedAgentRuntime()

async def main():
    await MyAgent.register(runtime, "my_agent", lambda: MyAgent())
    runtime.start()  # Start processing messages in the background.
    await runtime.send_message(MyMessageType("Hello, World!"), AgentId("my_agent", "default"))
    await runtime.stop()  # Stop processing messages in the background.

if __name__ == "__main__":
    asyncio.run(main())
