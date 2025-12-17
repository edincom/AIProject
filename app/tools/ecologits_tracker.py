# app/tools/ecologits_tracker.py
"""
Tracker centralisé pour les impacts environnementaux
Permet d'accumuler et d'afficher les statistiques globales
"""
import json
import os
from datetime import datetime
from typing import Dict, Any, List

class EcoLogitsTracker:
    """Tracker singleton pour accumuler les impacts"""
    
    _instance = None
    _impacts_file = "ecologits_impacts.json"
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self._initialized = True
        self.session_impacts = []
        self.load_history()
    
    def load_history(self):
        """Charge l'historique des impacts depuis le fichier"""
        if os.path.exists(self._impacts_file):
            try:
                with open(self._impacts_file, 'r') as f:
                    self.history = json.load(f)
            except:
                self.history = []
        else:
            self.history = []
    
    def save_history(self):
        """Sauvegarde l'historique dans un fichier"""
        with open(self._impacts_file, 'w') as f:
            json.dump(self.history, f, indent=2)
    
    def record_impact(self, 
                     mode: str,
                     username: str,
                     energy: float,
                     gwp: float,
                     model: str,
                     operation: str = "chat"):
        """
        Enregistre un impact
        
        Args:
            mode: "teach" ou "test"
            username: nom de l'utilisateur
            energy: consommation énergétique en Wh
            gwp: émissions GES en kgCO2eq
            model: nom du modèle utilisé
            operation: type d'opération (chat, generate_question, grade)
        """
        impact_record = {
            "timestamp": datetime.now().isoformat(),
            "mode": mode,
            "username": username,
            "energy_wh": energy,
            "gwp_kgco2eq": gwp,
            "model": model,
            "operation": operation
        }
        
        self.session_impacts.append(impact_record)
        self.history.append(impact_record)
        self.save_history()
    
    def get_session_stats(self) -> Dict[str, Any]:
        """Retourne les statistiques de la session en cours"""
        if not self.session_impacts:
            return {
                "total_requests": 0,
                "total_energy_wh": 0.0,
                "total_gwp_kgco2eq": 0.0
            }
        
        total_energy = sum(imp["energy_wh"] for imp in self.session_impacts)
        total_gwp = sum(imp["gwp_kgco2eq"] for imp in self.session_impacts)
        
        return {
            "total_requests": len(self.session_impacts),
            "total_energy_wh": total_energy,
            "total_gwp_kgco2eq": total_gwp,
            "avg_energy_per_request": total_energy / len(self.session_impacts),
            "avg_gwp_per_request": total_gwp / len(self.session_impacts)
        }
    
    def get_user_stats(self, username: str) -> Dict[str, Any]:
        """Retourne les statistiques pour un utilisateur spécifique"""
        user_impacts = [imp for imp in self.history if imp["username"] == username]
        
        if not user_impacts:
            return {
                "total_requests": 0,
                "total_energy_wh": 0.0,
                "total_gwp_kgco2eq": 0.0
            }
        
        total_energy = sum(imp["energy_wh"] for imp in user_impacts)
        total_gwp = sum(imp["gwp_kgco2eq"] for imp in user_impacts)
        
        # Équivalences concrètes
        km_driven = total_gwp * 5.5  # 1 kgCO2eq ≈ 5.5 km en voiture
        smartphone_charges = total_energy / 0.015  # 15 Wh par charge de smartphone
        
        return {
            "total_requests": len(user_impacts),
            "total_energy_wh": total_energy,
            "total_gwp_kgco2eq": total_gwp,
            "equivalents": {
                "km_driven": km_driven,
                "smartphone_charges": smartphone_charges
            }
        }
    
    def print_session_summary(self):
        """Affiche un résumé de la session"""
        stats = self.get_session_stats()
        
        if stats["total_requests"] == 0:
            print("\n📊 Aucune requête EcoLogits dans cette session")
            return
        
        print("\n" + "="*60)
        print("📊 RÉSUMÉ ECOLOGITS - SESSION")
        print("="*60)
        print(f"Total de requêtes: {stats['total_requests']}")
        print(f"Énergie totale: {stats['total_energy_wh']:.6f} Wh")
        print(f"Émissions GES totales: {stats['total_gwp_kgco2eq']:.9f} kgCO2eq")
        print(f"Moyenne par requête:")
        print(f"  - Énergie: {stats['avg_energy_per_request']:.6f} Wh")
        print(f"  - GES: {stats['avg_gwp_per_request']:.9f} kgCO2eq")
        print("="*60 + "\n")
    
    def print_user_summary(self, username: str):
        """Affiche un résumé pour un utilisateur"""
        stats = self.get_user_stats(username)
        
        if stats["total_requests"] == 0:
            print(f"\n📊 Aucune donnée pour l'utilisateur '{username}'")
            return
        
        print("\n" + "="*60)
        print(f"📊 RÉSUMÉ ECOLOGITS - Utilisateur: {username}")
        print("="*60)
        print(f"Total de requêtes: {stats['total_requests']}")
        print(f"Énergie totale: {stats['total_energy_wh']:.6f} Wh")
        print(f"Émissions GES totales: {stats['total_gwp_kgco2eq']:.9f} kgCO2eq")
        print(f"\n🌍 Équivalences concrètes:")
        print(f"  🚗 Distance en voiture: {stats['equivalents']['km_driven']:.2f} km")
        print(f"  📱 Charges de smartphone: {stats['equivalents']['smartphone_charges']:.1f}")
        print("="*60 + "\n")


# Instance globale
tracker = EcoLogitsTracker()