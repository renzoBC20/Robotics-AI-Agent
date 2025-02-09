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

    @message_handler
    async def handle_my_message_type(self, message: MyMessageType, ctx: MessageContext) -> MyMessageType:
        if message.content == "Iniciar":
            # Primera iteración, solo enviar temperatura inicial
            pos_actual = self.obtener_posicion_robot()
            temp = self.calcular_temperatura(pos_actual['x'], pos_actual['y'])
            print(f"\nPosición actual del robot: ({self.robot_x}, {self.robot_y})")
            print(f"Temperatura actual: {temp}°C")
            return MyMessageType(str(temp))
        else:
            # Procesar la respuesta del robot y calcular nueva temperatura
            parsed = self.parsear_respuesta_robot(message.content)
            self.mover_robot(parsed['movx'], parsed['movy'])
            print(f"Razonamiento del robot: {parsed['razonamiento']}")
            print(f"Movimiento realizado: ({parsed['movx']}, {parsed['movy']})")
            
            # Verificar si llegó a la fuente de calor
            if abs(self.robot_x - self.fuente_calor_x) < 0.5 and abs(self.robot_y - self.fuente_calor_y) < 0.5:
                print("\n¡El robot ha encontrado la fuente de calor!")
                return MyMessageType("FIN")
            
            # Calcular y enviar nueva temperatura
            temp = self.calcular_temperatura(self.robot_x, self.robot_y)
            print(f"Nueva posición: ({self.robot_x}, {self.robot_y})")
            print(f"Nueva temperatura: {temp}°C")
            return MyMessageType(str(temp))

class Robot(RoutedAgent):
    def __init__(self) -> None:
        super().__init__("MyAgent")
        self.system_message = SystemMessage(content=system_prompt)
        self.chat_history: List[LLMMessage]=[self.system_message]

    @message_handler
    async def handle_my_message_type(self, message: MyMessageType, ctx: MessageContext) -> MyMessageType:
        if message.content == "FIN":
            return MyMessageType("FIN")
        
        try:
            # Procesar la temperatura recibida
            temperatura = float(message.content)
            
            # Enviar la temperatura al modelo
            user_message = UserMessage(
                content=f"La temperatura actual es {temperatura}°C",
                source="user"
            )
            self.chat_history.append(user_message)
            
            # Obtener decisión del modelo
            response = await model_client.create(self.chat_history)
            assistant_message = AssistantMessage(
                content=response.content,
                source="assistant"
            )
            self.chat_history.append(assistant_message)
            
            # Retornar la decisión de movimiento
            return MyMessageType(response.content)
            
        except ValueError:
            print(f"Error: temperatura inválida recibida: {message.content}")
            return MyMessageType("FIN")

runtime = SingleThreadedAgentRuntime()

async def main():
    await Robot.register(runtime, "Robot", lambda: Robot())
    await Simulador.register(runtime, "Simulador", lambda: Simulador())
    runtime.start()  # Start processing messages in the background.

    print("\nIniciando simulación de búsqueda de fuente de calor...")
    print(f"Fuente de calor ubicada en (8, 10)")
    print("El robot comenzará en (0,0)")
    
    # Iniciar el ciclo de simulación
    mensaje_actual = MyMessageType("Iniciar")
    max_iteraciones = 20  # Límite de iteraciones para evitar bucles infinitos
    iteracion = 0
    
    while iteracion < max_iteraciones:
        iteracion += 1
        print(f"\n=== Iteración {iteracion} ===")
        
        # El simulador procesa el movimiento (si hay) y envía la temperatura
        respuesta_sim = await runtime.send_message(mensaje_actual, AgentId("Simulador", "default"))
        if respuesta_sim.content == "FIN":
            break
            
        # El robot recibe la temperatura y decide el movimiento
        respuesta_robot = await runtime.send_message(respuesta_sim, AgentId("Robot", "default"))
        if respuesta_robot.content == "FIN":
            break
            
        mensaje_actual = respuesta_robot
        await asyncio.sleep(2)  # Pausa entre iteraciones
    
    if iteracion >= max_iteraciones:
        print("\n¡Límite de iteraciones alcanzado! El robot no encontró la fuente de calor.")
    
    await runtime.stop()  # Stop processing messages in the background.

if __name__ == "__main__":
    asyncio.run(main())