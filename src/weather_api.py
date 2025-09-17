"""
🌤️ Module d'intégration API météorologique
Fonctionnalités:
- Récupération données météo historiques et actuelles
- Cache intelligent pour éviter surcoûts API
- Enrichissement automatique des datasets
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import json
import time
import logging
from typing import Dict, List, Optional
from dotenv import load_dotenv
import os

# Charger variables d'environnement
try:
    load_dotenv('config/api_keys.env', encoding='utf-8')
except UnicodeDecodeError:
    try:
        load_dotenv('config/api_keys.env', encoding='windows-1252')
    except:
        load_dotenv('config/api_keys.env', encoding='latin-1')

class WeatherEnrichment:
    """Enrichissement météorologique automatique"""
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.api_key = os.getenv('OPENWEATHER_API_KEY')
        self.base_url = "http://api.openweathermap.org/data/2.5"
        self.cache_path = Path('data/external/weather_cache.json')
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Paramètres Abidjan par défaut
        self.default_location = {
            'lat': 5.3600,
            'lon': -4.0083,
            'city': 'Abidjan',
            'country': 'CI'
        }
        
        # Charger cache
        self.weather_cache = self._load_cache()
        
        if not self.api_key:
            self.logger.warning("⚠️ Clé API OpenWeather manquante, utilisation données simulées")
    
    def _load_cache(self) -> Dict:
        """Charger cache météo existant"""
        try:
            if self.cache_path.exists():
                with open(self.cache_path, 'r') as f:
                    return json.load(f)
        except Exception as e:
            self.logger.warning(f"Erreur chargement cache: {e}")
        
        return {}
    
    def _save_cache(self):
        """Sauvegarder cache météo"""
        try:
            with open(self.cache_path, 'w') as f:
                json.dump(self.weather_cache, f, indent=2)
        except Exception as e:
            self.logger.error(f"Erreur sauvegarde cache: {e}")
    
    def get_historical_weather(self, start_date: str, end_date: str, 
                             location: Optional[Dict] = None) -> pd.DataFrame:
        """Récupérer données météo historiques"""
        self.logger.info(f"🌤️ Récupération météo {start_date} à {end_date}")
        
        if location is None:
            location = self.default_location
        
        # Générer liste des mois à traiter
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        date_range = pd.date_range(start=start_dt, end=end_dt, freq='MS')  # Month Start
        
        weather_data = []
        
        for date in date_range:
            month_key = date.strftime('%Y-%m')
            
            # Vérifier cache
            if month_key in self.weather_cache:
                self.logger.debug(f"📦 Cache hit pour {month_key}")
                weather_data.append(self.weather_cache[month_key])
                continue
            
            # Récupérer via API ou simuler
            if self.api_key:
                month_weather = self._fetch_monthly_weather(date, location)
            else:
                month_weather = self._simulate_weather_data(date)
            
            # Ajouter au cache et aux résultats
            month_weather['mois'] = month_key
            self.weather_cache[month_key] = month_weather
            weather_data.append(month_weather)
            
            # Délai pour éviter rate limiting
            time.sleep(0.1)
        
        # Sauvegarder cache
        self._save_cache()
        
        return pd.DataFrame(weather_data)
    
    def _fetch_monthly_weather(self, date: pd.Timestamp, location: Dict) -> Dict:
        """Récupérer données météo via API pour un mois"""
        try:
            # Utiliser API Current Weather comme approximation
            # Pour données historiques précises, utiliser API payante
            url = f"{self.base_url}/weather"
            params = {
                'lat': location['lat'],
                'lon': location['lon'],
                'appid': self.api_key,
                'units': 'metric'
            }
            
            response = requests.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                return {
                    'temp_moyenne': data['main']['temp'],
                    'temp_max': data['main']['temp_max'],
                    'temp_min': data['main']['temp_min'],
                    'humidite': data['main']['humidity'],
                    'pression': data['main']['pressure'],
                    'description': data['weather'][0]['description'],
                    'source': 'openweather_api'
                }
            else:
                self.logger.warning(f"Erreur API: {response.status_code}")
                return self._simulate_weather_data(date)
                
        except Exception as e:
            self.logger.error(f"Erreur récupération météo: {e}")
            return self._simulate_weather_data(date)
    
    def _simulate_weather_data(self, date: pd.Timestamp) -> Dict:
        """Simuler données météo réalistes pour Abidjan"""
        month = date.month
        
        # Températures moyennes mensuelles Abidjan (°C)
        temp_monthly = {
            1: 28, 2: 29, 3: 30, 4: 30, 5: 29, 6: 27,
            7: 26, 8: 26, 9: 27, 10: 28, 11: 29, 12: 28
        }
        
        # Humidité moyenne par mois (%)
        humidity_monthly = {
            1: 75, 2: 70, 3: 75, 4: 80, 5: 85, 6: 90,
            7: 90, 8: 85, 9: 85, 10: 80, 11: 80, 12: 75
        }
        
        base_temp = temp_monthly.get(month, 28)
        base_humidity = humidity_monthly.get(month, 80)
        
        # Ajouter variation aléatoire
        temp_variation = np.random.normal(0, 2)
        humidity_variation = np.random.normal(0, 5)
        
        return {
            'temp_moyenne': round(base_temp + temp_variation, 1),
            'temp_max': round(base_temp + temp_variation + 3, 1),
            'temp_min': round(base_temp + temp_variation - 3, 1),
            'humidite': max(40, min(100, base_humidity + humidity_variation)),
            'pression': round(1013 + np.random.normal(0, 10), 1),
            'description': 'simulated_data',
            'source': 'simulation'
        }
    
    def enrich_dataset_with_weather(self, df: pd.DataFrame) -> pd.DataFrame:
        """Enrichir dataset avec données météo"""
        self.logger.info("🌤️ Enrichissement avec données météo...")
        
        if 'mois' not in df.columns:
            self.logger.error("Colonne 'mois' manquante pour enrichissement météo")
            return df
        
        # Déterminer plage de dates
        df['mois_dt'] = pd.to_datetime(df['mois'])
        start_date = df['mois_dt'].min().strftime('%Y-%m-%d')
        end_date = df['mois_dt'].max().strftime('%Y-%m-%d')
        
        # Récupérer données météo
        weather_df = self.get_historical_weather(start_date, end_date)
        
        # Merger avec dataset principal
        df_enriched = df.merge(
            weather_df, 
            left_on=df['mois_dt'].dt.strftime('%Y-%m'),
            right_on='mois',
            how='left'
        )
        
        # Nettoyer
        df_enriched = df_enriched.drop(['mois_dt'], axis=1)
        
        # Calculer variables météo dérivées
        df_enriched = self._add_weather_derived_features(df_enriched)
        
        self.logger.info(f"✅ {len(df)} lignes enrichies avec météo")
        
        return df_enriched
    
    def _add_weather_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ajouter variables météo dérivées"""
        if 'temp_moyenne' in df.columns:
            # Besoin de climatisation (température > seuil de confort)
            df['besoin_clim'] = np.maximum(0, df['temp_moyenne'] - 26)
            df['besoin_chauffage'] = np.maximum(0, 18 - df['temp_moyenne'])
            
            # Catégories de température
            df['temp_category'] = pd.cut(
                df['temp_moyenne'],
                bins=[0, 22, 26, 30, 50],
                labels=['frais', 'confortable', 'chaud', 'très_chaud']
            )
            
            # Jours de forte chaleur (approximation)
            df['jours_tres_chauds'] = (df['temp_max'] > 32).astype(int)
        
        if 'humidite' in df.columns:
            # Inconfort lié à l'humidité
            df['inconfort_humidite'] = np.where(
                df['humidite'] > 80, 
                df['humidite'] - 80, 
                0
            )
        
        # Saisons Côte d'Ivoire
        if 'mois' in df.columns:
            df['mois_dt'] = pd.to_datetime(df['mois'])
            df['mois_numero'] = df['mois_dt'].dt.month
            df['saison_seche'] = df['mois_numero'].isin([11, 12, 1, 2, 3, 4]).astype(int)
            df['saison_pluies'] = df['mois_numero'].isin([5, 6, 7, 8, 9, 10]).astype(int)
        
        return df
    
    def get_current_weather(self, location: Optional[Dict] = None) -> Dict:
        """Récupérer météo actuelle"""
        if location is None:
            location = self.default_location
        
        if self.api_key:
            try:
                url = f"{self.base_url}/weather"
                params = {
                    'lat': location['lat'],
                    'lon': location['lon'],
                    'appid': self.api_key,
                    'units': 'metric'
                }
                
                response = requests.get(url, params=params, timeout=10)
                
                if response.status_code == 200:
                    data = response.json()
                    return {
                        'temperature': data['main']['temp'],
                        'humidity': data['main']['humidity'],
                        'description': data['weather'][0]['description'],
                        'timestamp': datetime.now().isoformat(),
                        'source': 'openweather_current'
                    }
            except Exception as e:
                self.logger.error(f"Erreur météo actuelle: {e}")
        
        # Fallback simulation
        return {
            'temperature': 28 + np.random.normal(0, 2),
            'humidity': 80 + np.random.normal(0, 5),
            'description': 'simulated',
            'timestamp': datetime.now().isoformat(),
            'source': 'simulation'
        }