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
    model="gpt-4o-mini",
    api_key=openai_api_key,
    # api_key="sk-...", # Optional if you have an API key set in the environment.
)


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
    def __init__(self,system_prompt) -> None:
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
        

async def ejecutar_simulacion(prompt: str, max_iteraciones: int) -> tuple[int, bool, List[Dict]]:
    runtime = SingleThreadedAgentRuntime()
    await Robot.register(runtime, "Robot", lambda: Robot(prompt))
    await Simulador.register(runtime, "Simulador", lambda: Simulador())
    runtime.start()

    print("\nIniciando simulación de búsqueda de fuente de calor...")
    print(f"Fuente de calor ubicada en (8, 10)")
    print("El robot comenzará en (0,0)")
    
    mensaje_actual = MyMessageType("Iniciar")
    iteracion = 0
    encontro_fuente = False
    historial_movimientos = []
    
    try:
        while iteracion < max_iteraciones:
            iteracion += 1
            print(f"\n=== Iteración {iteracion} ===")
            
            # Obtener temperatura actual
            respuesta_sim = await runtime.send_message(mensaje_actual, AgentId("Simulador", "default"))
            if respuesta_sim.content == "FIN":
                print("\n¡El robot ha encontrado la fuente de calor!")
                encontro_fuente = True
                break
            
            temperatura_actual = float(respuesta_sim.content)
            
            # Obtener movimiento del robot
            respuesta_robot = await runtime.send_message(respuesta_sim, AgentId("Robot", "default"))
            if respuesta_robot.content == "FIN":
                break
            
            # Parsear respuesta del robot
            try:
                think_pattern = r'<think>(.*?)</think>'
                movx_pattern = r'<movx>(.*?)</movx>'
                movy_pattern = r'<movy>(.*?)</movy>'
                
                razonamiento = re.search(think_pattern, respuesta_robot.content)
                movx = re.search(movx_pattern, respuesta_robot.content)
                movy = re.search(movy_pattern, respuesta_robot.content)
                
                # Registrar movimiento
                movimiento = {
                    'iteracion': iteracion,
                    'temperatura': temperatura_actual,
                    'razonamiento': razonamiento.group(1) if razonamiento else 'No especificado',
                    'movimiento_x': float(movx.group(1)) if movx else 0.0,
                    'movimiento_y': float(movy.group(1)) if movy else 0.0
                }
                historial_movimientos.append(movimiento)
            except Exception as e:
                print(f"Error al parsear respuesta del robot: {e}")
                
            mensaje_actual = respuesta_robot
            await asyncio.sleep(2)
        
        if iteracion >= max_iteraciones:
            print("\n¡Límite de iteraciones alcanzado! El robot no encontró la fuente de calor.")
        
        return iteracion, encontro_fuente, historial_movimientos
    
    finally:
        await runtime.stop()

async def optimizar_prompt_con_gpt(prompt_actual: str, historial: List[Dict], iteraciones: int, encontro_fuente: bool, historial_analisis: str) -> tuple[str, str]:
    """
    Utiliza GPT para optimizar el prompt basado en el rendimiento de la simulación.
    """
    optimization_client = OpenAIChatCompletionClient(
        model="gpt-4o-mini",
        api_key=openai_api_key,
    )

    # Crear mensaje de análisis para GPT
    analisis = f"""
Analiza el siguiente historial de movimientos de un robot buscador de calor y mejora el prompt.

Información de la simulación:
- Posición de la fuente de calor: (8, 10)
- Iteraciones realizadas: {iteraciones}
- Encontró la fuente: {'Sí' if encontro_fuente else 'No'}

Historial de movimientos:
"""
    # Agregar cada movimiento al análisis
    for mov in historial:
        analisis += f"""
Iteración {mov['iteracion']}:
  Temperatura: {mov['temperatura']}°C
  Razonamiento: {mov['razonamiento']}
  Movimiento: ({mov['movimiento_x']}, {mov['movimiento_y']})
"""

    analisis += f"""
Prompt actual:
{prompt_actual}

Por favor, genera un nuevo prompt mejorado que:
1. Corrija patrones ineficientes observados en los movimientos
2. Mejore la estrategia de búsqueda basada en los cambios de temperatura
3. Optimice la velocidad de convergencia hacia la fuente
4. Mantenga el mismo formato de respuesta (<think>, <movx>, <movy>)
5. Incluya mejores heurísticas para determinar cuándo se ha encontrado la fuente
6. Aproveche mejor la información de temperatura para tomar decisiones más informadas

A continuación dispones del historial de intentos realizados hasta el momento y sus resulltados:
{historial_analisis}
Retorna SOLO el nuevo prompt, sin explicaciones adicionales.
"""

    messages = [
        SystemMessage(content="Eres un experto en optimización de estrategias de búsqueda para robots."),
        UserMessage(content=analisis, source="user")
    ]
    
    response = await optimization_client.create(messages)
    return response.content, analisis

async def main():
    # Crear archivo de registro si no existe
    with open('registro_prompts.txt', 'w', encoding='utf-8') as f:
        f.write("=== Registro de Prompts y Resultados ===\n\n")
    
    max_simulaciones = int(input("Ingrese el número máximo de simulaciones: "))
    prompt_actual = input("Ingrese el prompt inicial para el robot: ")
    max_iteraciones = int(input("Ingrese la cantidad máxima de iteraciones: "))

    mejor_prompt = None
    mejor_rendimiento = float('inf')
    historial_analisis = ""

    for sim in range(max_simulaciones):
        print(f"\n{'='*50}")
        print(f"Simulación {sim + 1}/{max_simulaciones}")
        print(f"{'='*50}")
        
        # Ejecutar simulación
        iteraciones, encontro_fuente, historial = await ejecutar_simulacion(prompt_actual, max_iteraciones)
        
        # Registrar resultados
        with open('registro_prompts.txt', 'a', encoding='utf-8') as f:
            f.write(f"\nSimulación {sim + 1}:\n")
            f.write(f"Iteraciones: {iteraciones}\n")
            f.write(f"Encontró la fuente: {'Sí' if encontro_fuente else 'No'}\n")
            f.write("Prompt utilizado:\n")
            f.write(f"{prompt_actual}\n")
            f.write("\nHistorial de movimientos:\n")
            for mov in historial:
                f.write(f"\nIteración {mov['iteracion']}:\n")
                f.write(f"  Temperatura: {mov['temperatura']}°C\n")
                f.write(f"  Razonamiento: {mov['razonamiento']}\n")
                f.write(f"  Movimiento: ({mov['movimiento_x']}, {mov['movimiento_y']})\n")
            f.write("-" * 50 + "\n")
        
        # Actualizar mejor prompt si es necesario
        if encontro_fuente and iteraciones < mejor_rendimiento:
            mejor_rendimiento = iteraciones
            mejor_prompt = prompt_actual

        # Preguntar si desea optimizar el prompt con GPT
        if sim < max_simulaciones - 1:  # No preguntar en la última simulación
            print("\nOptimizando prompt con GPT...")
            prompt_actual, analisis = await optimizar_prompt_con_gpt(prompt_actual, historial, iteraciones, encontro_fuente, historial_analisis)
            historial_analisis += analisis
            print("\nNuevo prompt:")
            print(prompt_actual)

    
    # Registrar resumen final
    with open('registro_prompts.txt', 'a', encoding='utf-8') as f:
        f.write("\n=== Resumen Final ===\n")
        if mejor_prompt:
            f.write(f"Mejor rendimiento: {mejor_rendimiento} iteraciones\n")
            f.write("Mejor prompt:\n")
            f.write(f"{mejor_prompt}\n")
        else:
            f.write("Ninguna simulación encontró la fuente de calor.\n")
    
    print("\n=== Resultados guardados en 'registro_prompts.txt' ===")
    if mejor_prompt:
        print(f"Mejor rendimiento: {mejor_rendimiento} iteraciones")

if __name__ == "__main__":
    asyncio.run(main())