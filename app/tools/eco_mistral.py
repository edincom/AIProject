# app/tools/eco_mistral.py
"""
Wrapper simple pour tracker l'impact environnemental avec EcoLogits.
Usage: Remplace ChatMistralAI() par EcoMistralChat() dans vos chains.
"""

from ecologits import EcoLogits
from mistralai import Mistral
import os
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatResult, ChatGeneration

# Initialiser EcoLogits une seule fois
EcoLogits.init(providers=["mistralai"])

class EcoMistralChat(BaseChatModel):
    """
    Wrapper compatible LangChain qui utilise le SDK Mistral natif
    pour tracker les impacts environnementaux avec EcoLogits.
    """
    
    # Déclaration des champs Pydantic (requis par LangChain)
    model: str = "mistral-small-latest"
    temperature: float = 1.0
    streaming: bool = False
    
    # Attributs non-Pydantic (préfixés par _)
    _client: Mistral = None
    _total_co2: float = 0.0
    _total_water: float = 0.0
    _total_input_tokens: int = 0
    _total_output_tokens: int = 0
    _call_count: int = 0
    
    def __init__(self, model="mistral-small-latest", temperature=1, streaming=False, **kwargs):
        # Appeler le constructeur parent avec les champs Pydantic
        super().__init__(model=model, temperature=temperature, streaming=streaming, **kwargs)
        
        # Initialiser le client Mistral
        self._client = Mistral(api_key=os.getenv("MISTRAL_API_KEY"))
        
        # Initialiser les accumulateurs
        self._total_co2 = 0.0
        self._total_water = 0.0
        self._total_input_tokens = 0
        self._total_output_tokens = 0
        self._call_count = 0
    
    @property
    def _llm_type(self) -> str:
        """Retourne le type de LLM pour LangChain"""
        return "eco_mistral_chat"
    
    def _convert_langchain_messages(self, messages):
        """Convertit les messages LangChain en format Mistral SDK"""
        converted = []
        for msg in messages:
            if isinstance(msg, HumanMessage):
                role = "user"
                content = msg.content
            elif isinstance(msg, AIMessage):
                role = "assistant"
                content = msg.content
            elif isinstance(msg, SystemMessage):
                role = "system"
                content = msg.content
            elif hasattr(msg, 'type'):
                role = msg.type if msg.type in ['user', 'assistant', 'system'] else 'user'
                content = msg.content if hasattr(msg, 'content') else str(msg)
            else:
                # Format dict direct
                role = msg.get('role', 'user')
                content = msg.get('content', str(msg))
            
            converted.append({"role": role, "content": content})
        return converted
    
    def _update_stats(self, response):
        """Met à jour les statistiques environnementales"""
        if hasattr(response, 'impacts') and response.impacts:
            impacts = response.impacts
            
            # Extraction des données environnementales
            energy = getattr(impacts, 'energy', None)
            if energy:
                co2 = getattr(energy, 'value', 0)
                self._total_co2 += co2
            
            water = getattr(impacts, 'gwp', None)  # Global Warming Potential
            if water:
                water_val = getattr(water, 'value', 0)
                self._total_water += water_val
            
            # Usage des tokens
            if hasattr(response, 'usage'):
                usage = response.usage
                self._total_input_tokens += getattr(usage, 'prompt_tokens', 0)
                self._total_output_tokens += getattr(usage, 'completion_tokens', 0)
            
            self._call_count += 1
            
            # Log immédiat pour cette requête (SANS arrondir)
            print(f"\n{'='*60}")
            print(f"🌍 IMPACT ENVIRONNEMENTAL (Appel #{self._call_count})")
            print(f"{'='*60}")
            if energy:
                print(f"💨 CO2: {co2:.6f} gCO2eq")  # 6 décimales
            if water:
                print(f"💧 Eau: {water_val:.6f} L")  # 6 décimales
            if hasattr(response, 'usage'):
                print(f"📊 Tokens: {getattr(response.usage, 'prompt_tokens', 0)} in / {getattr(response.usage, 'completion_tokens', 0)} out")
            print(f"{'='*60}\n")
    
    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        """
        Méthode principale requise par LangChain BaseChatModel.
        Génère une réponse à partir des messages.
        """
        # Convertir les messages
        mistral_messages = self._convert_langchain_messages(messages)
        
        # Appel Mistral SDK avec tracking EcoLogits
        response = self._client.chat.complete(
            model=self.model,
            messages=mistral_messages,
            temperature=self.temperature
        )
        
        # Tracker les impacts
        self._update_stats(response)
        
        # Retourner au format LangChain
        content = response.choices[0].message.content
        message = AIMessage(content=content)
        generation = ChatGeneration(message=message)
        
        return ChatResult(generations=[generation])
    
    def _stream(self, messages, stop=None, run_manager=None, **kwargs):
        """
        Méthode de streaming requise par LangChain BaseChatModel.
        """
        # Convertir les messages
        mistral_messages = self._convert_langchain_messages(messages)
        
        # Stream depuis Mistral SDK
        stream_response = self._client.chat.stream(
            model=self.model,
            messages=mistral_messages,
            temperature=self.temperature
        )
        
        full_content = ""
        
        for chunk in stream_response:
            if chunk.data.choices:
                delta = chunk.data.choices[0].delta.content
                if delta:
                    full_content += delta
                    
                    # Yield un ChatGenerationChunk compatible LangChain
                    from langchain_core.messages import AIMessageChunk
                    from langchain_core.outputs import ChatGenerationChunk
                    
                    message_chunk = AIMessageChunk(content=delta, id=str(self._call_count))
                    generation_chunk = ChatGenerationChunk(message=message_chunk)
                    yield generation_chunk
        
        # Récupérer les impacts après le stream
        try:
            final_response = self._client.chat.complete(
                model=self.model,
                messages=mistral_messages + [{"role": "assistant", "content": full_content}],
                temperature=self.temperature,
                max_tokens=1
            )
            self._update_stats(final_response)
        except:
            pass
    
    def get_stats_summary(self):
        """Retourne un résumé des impacts totaux (format lisible, pas de notation scientifique)"""
        return {
            "total_calls": self._call_count,
            "total_co2_grams": float(f"{self._total_co2:.8f}"),  # Format lisible
            "total_water_liters": float(f"{self._total_water:.8f}"),  # Format lisible
            "total_input_tokens": self._total_input_tokens,
            "total_output_tokens": self._total_output_tokens,
            "avg_co2_per_call": float(f"{self._total_co2 / max(1, self._call_count):.8f}"),  # Format lisible
            "avg_tokens_per_call": float(f"{(self._total_input_tokens + self._total_output_tokens) / max(1, self._call_count):.2f}")
        }
    
    def print_summary(self):
        """Affiche un joli résumé des impacts (SANS arrondir)"""
        stats = self.get_stats_summary()
        print(f"\n{'='*60}")
        print(f"🌍 RÉSUMÉ IMPACT ENVIRONNEMENTAL TOTAL")
        print(f"{'='*60}")
        print(f"📞 Nombre d'appels: {stats['total_calls']}")
        print(f"💨 CO2 total: {stats['total_co2_grams']:.6f} gCO2eq")  # 6 décimales
        print(f"💧 Eau totale: {stats['total_water_liters']:.6f} L")  # 6 décimales
        print(f"📊 Tokens totaux: {stats['total_input_tokens']} in / {stats['total_output_tokens']} out")
        print(f"📈 Moyenne CO2/appel: {stats['avg_co2_per_call']:.6f} gCO2eq")  # 6 décimales
        print(f"📈 Moyenne tokens/appel: {stats['avg_tokens_per_call']:.2f}")
        print(f"{'='*60}\n")