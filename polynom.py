from dataclasses import dataclass
from autogen_core import SingleThreadedAgentRuntime
from autogen_core import AgentId, MessageContext, RoutedAgent, message_handler
import asyncio
from math import prod

@dataclass
class Operands:
    content: list[int]

class SumAgent(RoutedAgent):
    def __init__(self) -> None:
        super().__init__("SumAgent")

    @message_handler
    async def handle_operands(self, message: Operands, ctx: MessageContext) -> float:
        suma = float(sum(message.content))
        return suma

class MultiplyAgent(RoutedAgent):
    def __init__(self) -> None:
        super().__init__("MultiplyAgent")

    @message_handler
    async def handle_operands(self, message: Operands, ctx: MessageContext) -> float:
        producto = float(prod(message.content))
        return producto
    
class PowerAgent(RoutedAgent):
    def __init__(self) -> None:
        super().__init__("PowerAgent")

    @message_handler
    async def handle_operands(self, message: Operands, ctx: MessageContext) -> float:
        exp = float(pow(message.content[0], message.content[1]))
        return exp
    
@dataclass
class Entrada:
    content: float

class LeaderAgent(RoutedAgent):
    def __init__(self) -> None:
        super().__init__("LeaderAgent")

    @message_handler
    async def handle_operands(self, message: Entrada, ctx: MessageContext) -> None:
        # Convertir el input a float
        x = float(message.content)
        
        # Calcular x^3
        factor1 = await self.send_message(Operands([x, 3]), AgentId("PowerAgent","default"))
        
        # Calcular x^2
        factor2 = await self.send_message(Operands([x, 2]), AgentId("PowerAgent","default"))
        
        # Calcular -5x
        prod1 = await self.send_message(Operands([x, -5]), AgentId("MultiplyAgent","default"))
        
        # Calcular 3x^2
        prod2 = await self.send_message(Operands([factor2, 3]), AgentId("MultiplyAgent","default"))
        
        # Sumar todo: x^3 - 5x + 3x^2 + 10
        result = await self.send_message(Operands([factor1, prod1, prod2, 10]), AgentId("SumAgent","default"))
        
        print(f"El resultado del polinomio es: {result}")

runtime = SingleThreadedAgentRuntime()

async def main():
    await SumAgent.register(runtime, "SumAgent", lambda: SumAgent())
    await MultiplyAgent.register(runtime, "MultiplyAgent", lambda: MultiplyAgent())
    await PowerAgent.register(runtime, "PowerAgent", lambda: PowerAgent())
    await LeaderAgent.register(runtime, "LeaderAgent", lambda: LeaderAgent())
    runtime.start()  # Start processing messages in the background.
    mensaje = input("Ingrese el valor a calcular: ")
    await runtime.send_message(Entrada(mensaje), AgentId("LeaderAgent", "default"))
    await runtime.stop()  # Stop processing messages in the background.

if __name__ == "__main__":
    asyncio.run(main())