"""
🔮 Module de prédiction énergétique
Prédictions, scénarios et calculs d'économies
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Optional, Any, Union
import json
try:
    from cie_tarification import CIETarificationSystem, calculate_switching_savings
    from cie_ml_integration import CIEMLIntegrator, CIEClientAdvisor
    CIE_AVAILABLE = True
except ImportError:
    CIE_AVAILABLE = False
    print("⚠️ Module CIE non disponible")

class EnergyPredictor:
    """Prédicteur énergétique intelligent"""
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.models_path = Path('models')
        self.loaded_models = {}
        
        # Paramètres par secteur
        self.sector_defaults = {
            'residential': {
                'surface_m2': 120,
                'nb_personnes': 4,
                'base_consumption': 800
            },
            'office': {
                'surface_m2': 200,
                'nb_employes': 10,
                'base_consumption': 2000
            },
            'retail': {
                'surface_m2': 150,
                'base_consumption': 1500
            },
            'hotel': {
                'nb_chambres': 15,
                'surface_m2': 500,
                'taux_occupation': 0.70,
                'base_consumption': 4000
            }
        }
        
        # Tarifs électriques
        self.electricity_prices = {
            'residential': 140,  # FCFA/kWh
            'office': 150,
            'retail': 155,
            'hotel': 160
        }
    
    def load_model(self, model_path: Union[str, Path], model_name: str = None):
        """Charger modèle entraîné"""
        
        if isinstance(model_path, str):
            model_path = Path(model_path)
        
        if not model_path.exists():
            # Essayer de trouver le modèle le plus récent
            if model_name:
                pattern = f"{model_name}_model_v*.pkl"
                model_files = list(self.models_path.glob(pattern))
                if model_files:
                    model_path = max(model_files, key=lambda x: x.stat().st_mtime)
                else:
                    raise FileNotFoundError(f"Aucun modèle {model_name} trouvé")
        
        self.logger.info(f"🔄 Chargement modèle: {model_path}")
        
        model_info = joblib.load(model_path)
        model_key = model_name or model_path.stem
        
        self.loaded_models[model_key] = model_info
        self.logger.info(f"✅ Modèle {model_key} chargé")
        
        return model_key
    
    def load_best_model(self, metric: str = 'test_r2') -> str:
        """Charger le meilleur modèle basé sur une métrique"""
        
        metadata_file = self.models_path / 'model_metadata.json'
        
        if not metadata_file.exists():
            # Fallback: charger le modèle le plus récent disponible
            model_files = list(self.models_path.glob('*_model_v*.pkl'))
            if model_files:
                latest_model = max(model_files, key=lambda x: x.stat().st_mtime)
                return self.load_model(latest_model)
            else:
                raise FileNotFoundError("Aucun modèle trouvé")
        
        with open(metadata_file, 'r') as f:
            all_metadata = json.load(f)
        
        # Trouver le meilleur modèle
        best_score = -float('inf') if 'r2' in metric else float('inf')
        best_model = None
        best_filename = None
        
        for model_type, model_list in all_metadata.items():
            for model_data in model_list:
                if metric in model_data:
                    score = model_data[metric]
                    
                    if ('r2' in metric and score > best_score) or \
                       ('mae' in metric.lower() and score < best_score):
                        best_score = score
                        best_model = model_type
                        best_filename = model_data['filename']
        
        if best_model:
            model_path = self.models_path / best_filename
            model_key = self.load_model(model_path, best_model)
            self.logger.info(f"🏆 Meilleur modèle chargé: {best_model} ({metric}={best_score:.3f})")
            return model_key
        else:
            raise ValueError(f"Aucun modèle trouvé avec métrique {metric}")
    
    def predict_single(self, features: Union[List, np.ndarray, pd.DataFrame], 
                      model_key: str = None) -> float:
        """Prédiction pour un cas unique"""
        
        if model_key is None:
            if len(self.loaded_models) == 1:
                model_key = list(self.loaded_models.keys())[0]
            else:
                raise ValueError("Spécifiez model_key ou chargez un seul modèle")
        
        if model_key not in self.loaded_models:
            raise ValueError(f"Modèle {model_key} non chargé")
        
        model_info = self.loaded_models[model_key]
        
        # Gérer ensemble vs modèle simple
        if 'base_models' in model_info:
            # Modèle ensemble
            predictions = []
            for model_name, model in model_info['base_models']:
                pred = model.predict(np.array(features).reshape(1, -1))[0]
                predictions.append(pred)
            
            # Moyenne pondérée
            weights = model_info['weights']
            prediction = np.average(predictions, weights=weights)
        else:
            # Modèle simple
            model = model_info['model']
            scaler = model_info.get('scaler')
            
            # Normalisation si nécessaire
            if scaler:
                features = scaler.transform(np.array(features).reshape(1, -1))
            
            prediction = model.predict(np.array(features).reshape(1, -1))[0]
        
        return max(0, prediction)  # Éviter prédictions négatives
    
    def predict_sector(self, sector: str, parameters: Dict[str, Any], 
                      horizon: str = '1month', model_key: str = None) -> Dict[str, Any]:
        """Méthode principale pour prédictions par secteur - Compatible avec main.py"""
        
        self.logger.info(f"🔮 Prédiction {sector} - horizon: {horizon}")
        
        # Charger modèle si nécessaire
        if model_key is None:
            try:
                model_key = self.load_best_model('test_r2')
            except:
                # Fallback: charger n'importe quel modèle disponible
                model_files = list(self.models_path.glob('*_model_v*.pkl'))
                if model_files:
                    latest_model = max(model_files, key=lambda x: x.stat().st_mtime)
                    model_key = self.load_model(latest_model)
                else:
                    raise FileNotFoundError("Aucun modèle ML disponible")
        
        # Paramètres par défaut du secteur
        defaults = self.sector_defaults.get(sector, self.sector_defaults['residential'])
        params = {**defaults, **parameters}
        
        if horizon == '1month':
            return self._predict_monthly(sector, params, model_key)
        elif horizon == '1year':
            return self._predict_annual(sector, params, model_key)
        else:
            raise ValueError(f"Horizon '{horizon}' non supporté. Utilisez '1month' ou '1year'")
    
    def _predict_monthly(self, sector: str, params: Dict[str, Any], model_key: str) -> Dict[str, Any]:
        """Prédiction mensuelle"""
        
        # Construire features pour le mois en cours
        current_month = datetime.now().month
        features = self._build_monthly_features(current_month, sector, params)
        
        # Faire prédiction
        predicted_kwh = self.predict_single(features, model_key)
        
        # Calculer coût
        price_per_kwh = self.electricity_prices.get(sector, 150)
        estimated_cost = predicted_kwh * price_per_kwh
        
        # Générer scénarios d'optimisation
        optimization_scenarios = self._generate_optimization_scenarios(
            sector, params, predicted_kwh, model_key
        )
        
        return {
            'horizon': '1month',
            'sector': sector,
            'parameters': params,
            'predicted_kwh': round(predicted_kwh, 1),
            'estimated_cost_fcfa': round(estimated_cost, 0),
            'price_per_kwh': price_per_kwh,
            'optimization_scenarios': optimization_scenarios,
            'prediction_range': {
                'min_kwh': round(predicted_kwh * 0.85, 1),
                'max_kwh': round(predicted_kwh * 1.15, 1)
            }
        }
    
    def _predict_annual(self, sector: str, params: Dict[str, Any], model_key: str) -> Dict[str, Any]:
        """Prédiction annuelle"""
        
        monthly_predictions = {}
        total_kwh = 0
        total_cost = 0
        
        # Noms des mois en français
        month_names = [
            'Janvier', 'Février', 'Mars', 'Avril', 'Mai', 'Juin',
            'Juillet', 'Août', 'Septembre', 'Octobre', 'Novembre', 'Décembre'
        ]
        
        for month in range(1, 13):
            # Construire features pour ce mois
            features = self._build_monthly_features(month, sector, params)
            
            # Prédiction mensuelle
            predicted_kwh = self.predict_single(features, model_key)
            
            # Coût mensuel
            price_per_kwh = self.electricity_prices.get(sector, 150)
            monthly_cost = predicted_kwh * price_per_kwh
            
            monthly_predictions[f'month_{month}'] = {
                'month_name': month_names[month-1],
                'predicted_kwh': round(predicted_kwh, 1),
                'estimated_cost_fcfa': round(monthly_cost, 0)
            }
            
            total_kwh += predicted_kwh
            total_cost += monthly_cost
        
        # Résumé annuel
        annual_summary = {
            'total_kwh': round(total_kwh, 1),
            'total_cost_fcfa': round(total_cost, 0),
            'average_monthly_kwh': round(total_kwh / 12, 1),
            'average_monthly_cost_fcfa': round(total_cost / 12, 0)
        }
        
        # Scénarios d'optimisation basés sur la consommation annuelle
        optimization_scenarios = self._generate_optimization_scenarios(
            sector, params, total_kwh, model_key, annual=True
        )
        
        return {
            'horizon': '1year',
            'sector': sector,
            'parameters': params,
            'monthly_predictions': monthly_predictions,
            'annual_summary': annual_summary,
            'optimization_scenarios': optimization_scenarios
        }
    
    def _build_monthly_features(self, month: int, sector: str, params: Dict[str, Any]) -> List[float]:
        """Construire features pour un mois donné"""
        
        features = []
        
        # Features temporelles
        features.append(month)  # mois de l'année
        features.append(np.sin(2 * np.pi * month / 12))  # cyclicité
        features.append(np.cos(2 * np.pi * month / 12))
        
        # Features sectorielles
        features.append(params.get('surface_m2', 150))
        
        if sector == 'residential':
            features.append(params.get('nb_personnes', 4))
            features.extend([0, 0, 0])  # padding pour autres secteurs
        elif sector == 'office':
            features.append(params.get('nb_employes', 10))
            features.extend([0, 0, 0])  # padding
        elif sector == 'hotel':
            features.append(params.get('nb_chambres', 15))
            features.append(params.get('taux_occupation', 0.7))
            features.extend([0, 0])  # padding
        else:  # retail
            features.extend([0, 0, 0, 0])  # padding
        
        # Features météorologiques moyennes (Abidjan)
        temp_monthly = {
            1: 28, 2: 29, 3: 30, 4: 30, 5: 29, 6: 27,
            7: 26, 8: 26, 9: 27, 10: 28, 11: 29, 12: 28
        }
        features.append(temp_monthly.get(month, 28))
        
        # Humidité moyenne
        humidity_monthly = {
            1: 80, 2: 82, 3: 83, 4: 84, 5: 85, 6: 86,
            7: 87, 8: 86, 9: 85, 10: 83, 11: 81, 12: 80
        }
        features.append(humidity_monthly.get(month, 82))
        
        return features
    
    def _generate_optimization_scenarios(self, sector: str, params: Dict[str, Any], 
                                       base_consumption: float, model_key: str, 
                                       annual: bool = False) -> Dict[str, Any]:
        """Générer scénarios d'optimisation énergétique"""
        
        scenarios = {}
        base_cost = base_consumption * self.electricity_prices.get(sector, 150)
        
        # Scénario 1: Optimisation de base (5-15% d'économies)
        basic_savings_percent = 0.10  # 10% d'économies
        basic_consumption = base_consumption * (1 - basic_savings_percent)
        basic_cost = basic_consumption * self.electricity_prices.get(sector, 150)
        basic_savings_annual = (base_cost - basic_cost) * (12 if not annual else 1)
        basic_investment = 300000  # 300k FCFA
        
        scenarios['basic_optimization'] = {
            'description': 'Optimisation éclairage LED + programmation',
            'predicted_kwh': round(basic_consumption, 1),
            'cost_fcfa': round(basic_cost, 0),
            'efficiency_gain_percent': f'{basic_savings_percent*100:.0f}%',
            'cost_saved_annual_fcfa': round(basic_savings_annual, 0),
            'investment_fcfa': basic_investment,
            'roi_years': round(basic_investment / basic_savings_annual, 1),
            'is_profitable': (basic_investment / basic_savings_annual) <= 3
        }
        
        # Scénario 2: Optimisation avancée (15-25% d'économies)
        advanced_savings_percent = 0.20  # 20% d'économies
        advanced_consumption = base_consumption * (1 - advanced_savings_percent)
        advanced_cost = advanced_consumption * self.electricity_prices.get(sector, 150)
        advanced_savings_annual = (base_cost - advanced_cost) * (12 if not annual else 1)
        advanced_investment = 800000  # 800k FCFA
        
        scenarios['advanced_optimization'] = {
            'description': 'Système de climatisation intelligent + IoT',
            'predicted_kwh': round(advanced_consumption, 1),
            'cost_fcfa': round(advanced_cost, 0),
            'efficiency_gain_percent': f'{advanced_savings_percent*100:.0f}%',
            'cost_saved_annual_fcfa': round(advanced_savings_annual, 0),
            'investment_fcfa': advanced_investment,
            'roi_years': round(advanced_investment / advanced_savings_annual, 1),
            'is_profitable': (advanced_investment / advanced_savings_annual) <= 4
        }
        
        # Scénario 3: Optimisation complète avec énergies renouvelables
        complete_savings_percent = 0.35  # 35% d'économies
        complete_consumption = base_consumption * (1 - complete_savings_percent)
        complete_cost = complete_consumption * self.electricity_prices.get(sector, 150)
        complete_savings_annual = (base_cost - complete_cost) * (12 if not annual else 1)
        complete_investment = 1500000  # 1.5M FCFA
        
        scenarios['complete_optimization'] = {
            'description': 'Solution complète + panneaux solaires',
            'predicted_kwh': round(complete_consumption, 1),
            'cost_fcfa': round(complete_cost, 0),
            'efficiency_gain_percent': f'{complete_savings_percent*100:.0f}%',
            'cost_saved_annual_fcfa': round(complete_savings_annual, 0),
            'investment_fcfa': complete_investment,
            'roi_years': round(complete_investment / complete_savings_annual, 1),
            'is_profitable': (complete_investment / complete_savings_annual) <= 5
        }
        
        return scenarios
    
    def predict_monthly_profile(self, sector: str, parameters: Dict[str, Any], 
                              model_key: str = None, months: int = 12) -> Dict[str, Dict]:
        """Prédire profil de consommation mensuel - Méthode legacy"""
        
        # Rediriger vers predict_sector pour compatibilité
        result = self.predict_sector(sector, parameters, '1year', model_key)
        return result['monthly_predictions']
    
    def calculate_savings(self, current_consumption: float, predicted_consumption: float, 
                         sector: str) -> Dict[str, float]:
        """Calculer économies potentielles"""
        
        current_cost = current_consumption * self.electricity_prices.get(sector, 150)
        predicted_cost = predicted_consumption * self.electricity_prices.get(sector, 150)
        
        savings_kwh = current_consumption - predicted_consumption
        savings_fcfa = current_cost - predicted_cost
        savings_percentage = (savings_kwh / current_consumption) * 100 if current_consumption > 0 else 0
        
        return {
            'economie_kwh': round(savings_kwh, 2),
            'economie_fcfa': round(savings_fcfa, 2),
            'pourcentage_economie': round(savings_percentage, 2),
            'consommation_actuelle': round(current_consumption, 2),
            'consommation_optimisee': round(predicted_consumption, 2),
            'cout_actuel': round(current_cost, 2),
            'cout_optimise': round(predicted_cost, 2)
        }
    
    def save_predictions(self, predictions: Dict[str, Any], output_path: str):
        """Sauvegarder prédictions dans fichier"""
        
        output_path = Path(output_path)
        
        if output_path.suffix.lower() == '.json':
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(predictions, f, indent=2, ensure_ascii=False)
        elif output_path.suffix.lower() == '.csv':
            # Convertir en DataFrame pour export CSV
            if predictions['horizon'] == '1year':
                df_data = []
                for month_key, month_data in predictions['monthly_predictions'].items():
                    df_data.append({
                        'Mois': month_data['month_name'],
                        'Consommation_kWh': month_data['predicted_kwh'],
                        'Coût_FCFA': month_data['estimated_cost_fcfa']
                    })
                df = pd.DataFrame(df_data)
                df.to_csv(output_path, index=False)
            else:
                # Pour prédiction mensuelle, créer un CSV simple
                df_data = [{
                    'Secteur': predictions['sector'],
                    'Consommation_kWh': predictions['predicted_kwh'],
                    'Coût_FCFA': predictions['estimated_cost_fcfa'],
                    'Horizon': predictions['horizon']
                }]
                df = pd.DataFrame(df_data)
                df.to_csv(output_path, index=False)
        
        self.logger.info(f"💾 Prédictions sauvées: {output_path}")
    
    def display_predictions(self, predictions: Dict[str, Any]):
        """Afficher prédictions de façon formatée"""
        
        print(f"\n{'='*50}")
        print(f"🔮 PRÉDICTIONS ÉNERGÉTIQUES - {predictions['sector'].upper()}")
        print(f"{'='*50}")
        
        if predictions['horizon'] == '1month':
            print(f"📅 Horizon: Mois prochain")
            print(f"⚡ Consommation prévue: {predictions['predicted_kwh']:,.1f} kWh")
            print(f"💰 Coût estimé: {predictions['estimated_cost_fcfa']:,.0f} FCFA")
            print(f"📊 Prix unitaire: {predictions['price_per_kwh']} FCFA/kWh")
            
        elif predictions['horizon'] == '1year':
            summary = predictions['annual_summary']
            print(f"📅 Horizon: Année complète")
            print(f"⚡ Consommation annuelle: {summary['total_kwh']:,.1f} kWh")
            print(f"💰 Coût annuel: {summary['total_cost_fcfa']:,.0f} FCFA")
            print(f"📊 Moyenne mensuelle: {summary['average_monthly_kwh']:,.1f} kWh")
        
        # Afficher scénarios d'optimisation
        if 'optimization_scenarios' in predictions:
            print(f"\n💡 SCÉNARIOS D'OPTIMISATION:")
            print(f"-" * 40)
            
            for scenario_name, scenario in predictions['optimization_scenarios'].items():
                status = "✅ Recommandé" if scenario['is_profitable'] else "⚠️ À évaluer"
                print(f"\n{status} {scenario['description']}")
                print(f"   💪 Économies: {scenario['efficiency_gain_percent']}")
                print(f"   💰 Investissement: {scenario['investment_fcfa']:,} FCFA")
                print(f"   ⏰ ROI: {scenario['roi_years']} ans")
                print(f"   💚 Économies/an: {scenario['cost_saved_annual_fcfa']:,} FCFA")
        
        print(f"\n{'='*50}")