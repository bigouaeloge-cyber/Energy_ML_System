"""
🔧 Module de feature engineering avancé
Création de variables sophistiquées pour maximiser performance ML
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Optional
from sklearn.preprocessing import StandardScaler, LabelEncoder
from scipy import stats

class AdvancedFeatureEngineer:
    """Générateur de features avancées pour énergie"""
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.scalers = {}
        self.label_encoders = {}
    
    def create_comprehensive_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Créer ensemble complet de features"""
        self.logger.info("🔧 Génération features avancées...")
        
        df_features = df.copy()
        
        # Features temporelles
        df_features = self._add_temporal_features(df_features)
        
        # Features cycliques
        df_features = self._add_cyclical_features(df_features)
        
        # Features de lag et moyennes mobiles
        df_features = self._add_lag_features(df_features)
        
        # Features météo dérivées
        df_features = self._add_weather_features(df_features)
        
        # Features sectorielles
        df_features = self._add_sector_features(df_features)
        
        # Features d'interaction
        df_features = self._add_interaction_features(df_features)
        
        # Features statistiques
        df_features = self._add_statistical_features(df_features)
        
        self.logger.info(f"✅ {len(df_features.columns)} features générées")
        
        return df_features
    
    def _add_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Features temporelles avancées"""
        if 'mois' not in df.columns:
            return df
        
        df['mois_dt'] = pd.to_datetime(df['mois'])
        
        # Extractions temporelles
        df['annee'] = df['mois_dt'].dt.year
        df['mois_numero'] = df['mois_dt'].dt.month
        df['trimestre'] = df['mois_dt'].dt.quarter
        df['semaine_annee'] = df['mois_dt'].dt.isocalendar().week
        
        # Variables binaires temporelles
        df['debut_annee'] = (df['mois_numero'] <= 2).astype(int)
        df['fin_annee'] = (df['mois_numero'] >= 11).astype(int)
        df['milieu_annee'] = ((df['mois_numero'] >= 5) & (df['mois_numero'] <= 8)).astype(int)
        
        # Tendance temporelle
        df['mois_depuis_debut'] = (df['mois_dt'] - df['mois_dt'].min()).dt.days // 30
        
        return df
    
    def _add_cyclical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Features cycliques (sine/cosine) pour capturer saisonnalité"""
        
        if 'mois_numero' in df.columns:
            # Cycle mensuel
            df['mois_sin'] = np.sin(2 * np.pi * df['mois_numero'] / 12)
            df['mois_cos'] = np.cos(2 * np.pi * df['mois_numero'] / 12)
            
            # Cycle trimestriel  
            df['trimestre_sin'] = np.sin(2 * np.pi * df['trimestre'] / 4)
            df['trimestre_cos'] = np.cos(2 * np.pi * df['trimestre'] / 4)
        
        return df
    
    def _add_lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Features de décalage temporel et moyennes mobiles"""
        
        if 'kwh_consommes' not in df.columns:
            return df
        
        # Trier par secteur et date
        if 'secteur' in df.columns:
            df = df.sort_values(['secteur', 'mois_dt'])
        else:
            df = df.sort_values('mois_dt')
        
        # Lags de consommation
        lag_periods = [1, 2, 3, 6, 12]  # 1, 2, 3, 6 mois, 1 an
        
        for lag in lag_periods:
            if 'secteur' in df.columns:
                df[f'kwh_lag_{lag}'] = df.groupby('secteur')['kwh_consommes'].shift(lag)
            else:
                df[f'kwh_lag_{lag}'] = df['kwh_consommes'].shift(lag)
        
        # Moyennes mobiles
        windows = [3, 6, 12]
        for window in windows:
            if 'secteur' in df.columns:
                df[f'kwh_ma_{window}'] = df.groupby('secteur')['kwh_consommes'].rolling(window).mean().reset_index(0, drop=True)
            else:
                df[f'kwh_ma_{window}'] = df['kwh_consommes'].rolling(window).mean()
        
        # Tendances
        df['kwh_diff_1'] = df['kwh_consommes'].diff()
        df['kwh_diff_12'] = df['kwh_consommes'].diff(12)  # Variation année précédente
        
        # Ratios
        df['ratio_vs_ma3'] = df['kwh_consommes'] / df['kwh_ma_3']
        df['ratio_vs_ma12'] = df['kwh_consommes'] / df['kwh_ma_12']
        
        return df
    
    def _add_weather_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Features météorologiques avancées"""
        
        weather_cols = ['temp_moyenne', 'temp_max', 'temp_min', 'humidite']
        
        for col in weather_cols:
            if col in df.columns:
                # Différences et tendances
                df[f'{col}_diff'] = df[col].diff()
                df[f'{col}_ma3'] = df[col].rolling(3).mean()
                
                # Extremes
                df[f'{col}_is_extreme'] = (
                    (df[col] > df[col].quantile(0.95)) | 
                    (df[col] < df[col].quantile(0.05))
                ).astype(int)
        
        # Indices de confort
        if 'temp_moyenne' in df.columns and 'humidite' in df.columns:
            # Indice de chaleur approximatif
            df['indice_chaleur'] = df['temp_moyenne'] + 0.1 * df['humidite']
            
            # Zone de confort
            df['zone_confort'] = (
                (df['temp_moyenne'] >= 22) & 
                (df['temp_moyenne'] <= 26) &
                (df['humidite'] <= 80)
            ).astype(int)
        
        return df
    
    def _add_sector_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Features spécifiques par secteur"""
        
        if 'secteur' not in df.columns:
            return df
        
        # Encoding secteur
        if 'secteur' not in self.label_encoders:
            self.label_encoders['secteur'] = LabelEncoder()
            df['secteur_encoded'] = self.label_encoders['secteur'].fit_transform(df['secteur'])
        else:
            df['secteur_encoded'] = self.label_encoders['secteur'].transform(df['secteur'])
        
        # Moyennes par secteur
        sector_means = df.groupby('secteur')['kwh_consommes'].mean()
        df['secteur_moyenne_kwh'] = df['secteur'].map(sector_means)
        
        # Ratio vs moyenne sectorielle
        df['ratio_vs_secteur_moyen'] = df['kwh_consommes'] / df['secteur_moyenne_kwh']
        
        # Features spécialisées par secteur
        for secteur in df['secteur'].unique():
            mask = df['secteur'] == secteur
            
            if secteur == 'hotel':
                # Features hôtellerie
                if 'nb_chambres' in df.columns:
                    df.loc[mask, 'kwh_par_chambre'] = df.loc[mask, 'kwh_consommes'] / df.loc[mask, 'nb_chambres']
                if 'taux_occupation' in df.columns:
                    df.loc[mask, 'kwh_occupe'] = df.loc[mask, 'kwh_consommes'] / df.loc[mask, 'taux_occupation']
            
            elif secteur == 'office':
                # Features bureau
                if 'nb_employes' in df.columns:
                    df.loc[mask, 'kwh_par_employe'] = df.loc[mask, 'kwh_consommes'] / df.loc[mask, 'nb_employes']
            
            elif secteur == 'residential':
                # Features résidentiel
                if 'nb_personnes' in df.columns:
                    df.loc[mask, 'kwh_par_personne'] = df.loc[mask, 'kwh_consommes'] / df.loc[mask, 'nb_personnes']
        
        return df
    
    def _add_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Features d'interaction entre variables"""
        
        # Interactions température * secteur
        if 'temp_moyenne' in df.columns and 'secteur_encoded' in df.columns:
            df['temp_x_secteur'] = df['temp_moyenne'] * df['secteur_encoded']
        
        # Interactions surface * température
        if 'surface_m2' in df.columns and 'temp_moyenne' in df.columns:
            df['surface_x_temp'] = df['surface_m2'] * df['temp_moyenne']
        
        # Interactions saisonnalité * consommation historique
        if 'mois_sin' in df.columns and 'kwh_lag_12' in df.columns:
            df['saison_x_hist'] = df['mois_sin'] * df['kwh_lag_12']
        
        return df
    
    def _add_statistical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Features statistiques avancées"""
        
        if 'kwh_consommes' not in df.columns:
            return df
        
        # Features par secteur si disponible
        group_col = 'secteur' if 'secteur' in df.columns else None
        
        if group_col:
            # Statistiques groupées
            grouped = df.groupby(group_col)['kwh_consommes']
            
            df['kwh_zscore'] = grouped.transform(lambda x: (x - x.mean()) / x.std())
            df['kwh_percentile'] = grouped.transform(lambda x: x.rank(pct=True))
            df['kwh_vs_median'] = df['kwh_consommes'] - grouped.transform('median')
        
        # Détection d'anomalies
        Q1 = df['kwh_consommes'].quantile(0.25)
        Q3 = df['kwh_consommes'].quantile(0.75)
        IQR = Q3 - Q1
        
        df['is_outlier'] = (
            (df['kwh_consommes'] < Q1 - 1.5 * IQR) |
            (df['kwh_consommes'] > Q3 + 1.5 * IQR)
        ).astype(int)
        
        # Volatilité (écart-type mobile)
        df['kwh_volatility'] = df['kwh_consommes'].rolling(6).std()
        
        return df
    
    def select_best_features(self, df: pd.DataFrame, target_col: str = 'kwh_consommes', 
                           method: str = 'correlation', top_k: int = 50) -> List[str]:
        """Sélectionner les meilleures features"""
        
        if target_col not in df.columns:
            self.logger.error(f"Colonne cible {target_col} introuvable")
            return []
        
        # Exclure colonnes non-numériques et cible
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        feature_cols = [col for col in numeric_cols if col != target_col]
        
        if method == 'correlation':
            # Sélection par corrélation
            correlations = df[feature_cols].corrwith(df[target_col]).abs().sort_values(ascending=False)
            selected_features = correlations.head(top_k).index.tolist()
            
        elif method == 'mutual_info':
            # Information mutuelle
            from sklearn.feature_selection import mutual_info_regression
            mi_scores = mutual_info_regression(df[feature_cols].fillna(0), df[target_col])
            mi_df = pd.DataFrame({'feature': feature_cols, 'score': mi_scores})
            selected_features = mi_df.nlargest(top_k, 'score')['feature'].tolist()
            
        elif method == 'variance':
            # Variance threshold
            from sklearn.feature_selection import VarianceThreshold
            selector = VarianceThreshold(threshold=0.01)
            selector.fit(df[feature_cols].fillna(0))
            selected_features = [feature_cols[i] for i in range(len(feature_cols)) if selector.get_support()[i]][:top_k]
        
        else:
            selected_features = feature_cols[:top_k]
        
        self.logger.info(f"✅ {len(selected_features)} features sélectionnées par méthode {method}")
        
        return selected_features