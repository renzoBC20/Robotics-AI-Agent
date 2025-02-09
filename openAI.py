from dataclasses import dataclass
from autogen_core import SingleThreadedAgentRuntime
from autogen_core import AgentId, MessageContext, RoutedAgent, message_handler
import asyncio
from autogen_core.models import UserMessage, AssistantMessage, LLMMessage, SystemMessage
from autogen_ext.models.openai import OpenAIChatCompletionClient
from dotenv import load_dotenv
import os
from typing import List, Any, Dict
import math
import re
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
creas que estes en la fuente de calor.
El formato de tus respuestas debe ser el siguiente:
<think>razonamiento</think>
<movx>x</movx>
<movy>y</movy>
donde:
- razonamiento es una explicacion corta del movimiento
- x,y es el vector de movimiento, un valor positivo en x significa derecha,
un negativo en x es izquierda, uno positivo en y es arriba y uno negativo en y es abajo
"""

@dataclass
class MyMessageType:
    content: str

class Simulador(RoutedAgent):
    def __init__(self) -> None:
        super().__init__("Simulador")
        # Posición de la fuente de calor
        self.fuente_calor_x = 8
        self.fuente_calor_y = 10
        self.temperatura_maxima = 100  # Temperatura en la fuente de calor
        # Posición actual del robot
        self.robot_x = 0
        self.robot_y = 0

    def mover_robot(self, delta_x: float, delta_y: float) -> None:
        """
        Actualiza la posición del robot según el movimiento indicado
        """
        self.robot_x += delta_x
        self.robot_y += delta_y

    def obtener_posicion_robot(self) -> Dict[str, float]:
        """
        Retorna la posición actual del robot
        """
        return {
            'x': self.robot_x,
            'y': self.robot_y
        }

    def parsear_respuesta_robot(self, respuesta: str) -> Dict[str, Any]:
        """
        Parsea la respuesta XML del robot y retorna un diccionario con los campos.
        La respuesta tiene el formato:
        <think>razonamiento</think>
        <movx>x</movx>
        <movy>y</movy>
        """
        think_pattern = r'<think>(.*?)</think>'
        movx_pattern = r'<movx>(.*?)</movx>'
        movy_pattern = r'<movy>(.*?)</movy>'
        
        think = re.search(think_pattern, respuesta)
        movx = re.search(movx_pattern, respuesta)
        movy = re.search(movy_pattern, respuesta)
        
        return {
            'razonamiento': think.group(1) if think else '',
            'movx': float(movx.group(1)) if movx else 0.0,
            'movy': float(movy.group(1)) if movy else 0.0
        }

    def calcular_temperatura(self, pos_x: float, pos_y: float) -> float:
        """
        Calcula la temperatura en una posición dada basada en la distancia a la fuente de calor.
        La temperatura disminuye exponencialmente con la distancia.
        """
        distancia = math.sqrt((pos_x - self.fuente_calor_x)**2 + (pos_y - self.fuente_calor_y)**2)
        temperatura = self.temperatura_maxima * math.exp(-0.5 * distancia)
        return round(temperatura, 2)

    def set_fuente_calor(self, x: float, y: float) -> None:
        """
        Establece la posición de la fuente de calor
        """
        self.fuente_calor_x = x
        self.fuente_calor_y = y

    @message_handler
    async def handle_my_message_type(self, message: MyMessageType, ctx: MessageContext) -> None:
        print(f"Iniciando simulacion")
        # Obtener la temperatura en la posición actual del robot
        pos_actual = self.obtener_posicion_robot()
        temp = self.calcular_temperatura(pos_actual['x'], pos_actual['y'])
        
        # Enviar la temperatura al robot y obtener su respuesta
        respuesta = await self.send_message(MyMessageType(str(temp)), AgentId("Robot", "default"))
        parsed = self.parsear_respuesta_robot(respuesta.content)
        
        # Actualizar la posición del robot según su movimiento
        self.mover_robot(parsed['movx'], parsed['movy'])
        
        # Imprimir información del movimiento
        print(f"Razonamiento: {parsed['razonamiento']}")
        print(f"Movimiento: ({parsed['movx']}, {parsed['movy']})")
        print(f"Nueva posición: ({self.robot_x}, {self.robot_y})")
        print(f"Temperatura actual: {temp}°C")

class Robot(RoutedAgent):
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
        return MyMessageType(response.content)

runtime = SingleThreadedAgentRuntime()

async def main():
    await Robot.register(runtime, "Robot", lambda: Robot())
    await Simulador.register(runtime, "Simulador", lambda: Simulador())
    runtime.start()  # Start processing messages in the background.

    await runtime.send_message(MyMessageType("Iniciar"), AgentId("Simulador", "default"))
    
    await runtime.stop()  # Stop processing messages in the background.

if __name__ == "__main__":
    asyncio.run(main())