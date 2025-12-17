# app/tools/ecologits_wrapper.py
"""
Wrapper pour intégrer EcoLogits avec Mistral AI
Compatible avec l'interface LangChain existante
"""
import os
from typing import Iterator, List, Dict, Any
from ecologits import EcoLogits
from mistralai import Mistral

# Initialiser EcoLogits au chargement du module
EcoLogits.init(providers=["mistralai"])

class MistralWithEcoLogits:
    """
    Wrapper autour du client Mistral officiel avec tracking EcoLogits.
    Interface compatible avec le code existant.
    """
    
    def __init__(self, model: str, temperature: float = 1.0, streaming: bool = False):
        self.model = model
        self.temperature = temperature
        self.streaming = streaming
        self.client = Mistral(api_key=os.getenv("MISTRAL_API_KEY"))
        self.last_impacts = None
        
    def invoke(self, messages: List[Dict[str, str]]) -> Any:
        """Mode non-streaming - retourne la réponse complète"""
        response = self.client.chat.complete(
            model=self.model,
            messages=messages,
            temperature=self.temperature
        )
        
        # Stocker les impacts
        self.last_impacts = response.impacts
        
        # Créer un objet compatible avec LangChain
        class Response:
            def __init__(self, content, impacts):
                self.content = content
                self.impacts = impacts
        
        return Response(response.choices[0].message.content, response.impacts)
    
    def stream(self, messages: List[Dict[str, str]]) -> Iterator:
        """Mode streaming - yield les chunks un par un"""
        stream = self.client.chat.stream(
            model=self.model,
            messages=messages,
            temperature=self.temperature
        )
        
        for chunk in stream:
            # Stocker les impacts cumulatifs
            if hasattr(chunk, 'data') and hasattr(chunk.data, 'impacts'):
                self.last_impacts = chunk.data.impacts
            
            # Yield un objet compatible avec LangChain
            if hasattr(chunk, 'data') and hasattr(chunk.data, 'choices'):
                delta = chunk.data.choices[0].delta
                if hasattr(delta, 'content') and delta.content:
                    class StreamChunk:
                        def __init__(self, content, impacts=None):
                            self.content = content
                            self.impacts = impacts
                    
                    yield StreamChunk(delta.content, self.last_impacts)
    
    def get_last_impacts(self) -> Dict[str, Any]:
        """Récupère les impacts de la dernière requête"""
        if self.last_impacts is None:
            return None
        
        # Convertir en dict pour faciliter l'affichage
        return {
            "energy": {
                "value": self.last_impacts.energy.value,
                "unit": self.last_impacts.energy.unit
            },
            "gwp": {
                "value": self.last_impacts.gwp.value,
                "unit": self.last_impacts.gwp.unit
            }
        }
    
    def print_impacts(self, prefix: str = ""):
        """Affiche les impacts de manière formatée"""
        impacts = self.get_last_impacts()
        if impacts:
            print(f"\n{prefix}🌱 ECOLOGITS - Impact environnemental:")
            print(f"   ⚡ Énergie: {impacts['energy']['value']:.6f} {impacts['energy']['unit']}")
            print(f"   🌍 Émissions GES: {impacts['gwp']['value']:.9f} {impacts['gwp']['unit']}")


def format_langchain_prompt_for_mistral(prompt_value) -> List[Dict[str, str]]:
    """
    Convertit un prompt LangChain au format Mistral
    
    Args:
        prompt_value: Objet prompt de LangChain avec .to_messages()
    
    Returns:
        Liste de messages au format Mistral
    """
    messages = []
    
    for msg in prompt_value.to_messages():
        role = msg.__class__.__name__.replace("Message", "").lower()
        if role == "human":
            role = "user"
        elif role == "ai":
            role = "assistant"
        
        messages.append({
            "role": role,
            "content": msg.content
        })
    
    return messages