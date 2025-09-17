"""
🤖 Module d'entraînement des modèles ML
Entraînement, optimisation et validation des modèles énergétiques
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
import json
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any

# ML Libraries
from sklearn.model_selection import train_test_split, TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
try:
    from catboost import CatBoostRegressor
except ImportError:
    CatBoostRegressor = None

# Optimization
import optuna
import joblib

# Modules personnalisés
from feature_engineering import AdvancedFeatureEngineer
from data_processing import DataProcessor

class EnergyModelTrainer:
    """Entraîneur de modèles ML pour optimisation énergétique"""
    
    def __init__(self, config_path: str = 'config/config.yaml'):
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Chemins et configuration
        self.models_path = Path('models')
        self.models_path.mkdir(exist_ok=True)
        
        # Modèles disponibles
        self.available_models = {
            'linear': LinearRegression(),
            'ridge': Ridge(alpha=1.0),
            'lasso': Lasso(alpha=1.0),
            'random_forest': RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            ),
            'xgboost': XGBRegressor(
                n_estimators=200,
                max_depth=8,
                learning_rate=0.1,
                random_state=42
            ),
            'lightgbm': LGBMRegressor(
                n_estimators=200,
                max_depth=8,
                learning_rate=0.1,
                random_state=42,
                verbose=-1
            ),
            'gradient_boosting': GradientBoostingRegressor(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                random_state=42
            )
        }
        
        # Ajouter CatBoost si disponible
        if CatBoostRegressor:
            self.available_models['catboost'] = CatBoostRegressor(
                iterations=200,
                depth=8,
                learning_rate=0.1,
                random_seed=42,
                verbose=False
            )
        
        # Feature engineer
        self.feature_engineer = AdvancedFeatureEngineer()
        
        # Scalers pour normalisation
        self.scalers = {}
    
    def load_and_prepare_data(self, data_path: str) -> Tuple[pd.DataFrame, List[str]]:
        """Charger et préparer données pour entraînement"""
        self.logger.info(f"📊 Chargement données: {data_path}")
        
        # Charger données
        df = pd.read_csv(data_path)
        self.logger.info(f"✅ {len(df)} lignes chargées")
        
        # Générer features avancées
        df_features = self.feature_engineer.create_comprehensive_features(df)
        
        # Sélectionner meilleures features
        feature_columns = self.feature_engineer.select_best_features(
            df_features, 
            target_col='kwh_consommes',
            method='correlation',
            top_k=30
        )
        
        self.logger.info(f"🎯 {len(feature_columns)} features sélectionnées")
        
        return df_features, feature_columns
    
    def prepare_train_test_split(self, df: pd.DataFrame, feature_columns: List[str], 
                               target_col: str = 'kwh_consommes',
                               test_size: float = 0.2) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Préparer split train/test avec validation temporelle"""
        
        # Préparer features et target
        X = df[feature_columns].fillna(0)
        y = df[target_col]
        
        # Supprimer outliers extrêmes
        Q1, Q3 = y.quantile([0.01, 0.99])
        mask = (y >= Q1) & (y <= Q3)
        X, y = X[mask], y[mask]
        
        # Split temporel si colonne date disponible
        if 'mois_dt' in df.columns:
            df_clean = df[mask].reset_index(drop=True)
            cutoff_date = df_clean['mois_dt'].quantile(1 - test_size)
            train_mask = df_clean['mois_dt'] <= cutoff_date
            
            train_indices = X.index[train_mask]
            test_indices = X.index[~train_mask]
            X_train, X_test = X.loc[train_indices], X.loc[test_indices]
            y_train, y_test = y.loc[train_indices], y.loc[test_indices]
        else:
            # Split aléatoire classique
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42
            )
        
        self.logger.info(f"📈 Split: {len(X_train)} train, {len(X_test)} test")
        
        return X_train, X_test, y_train, y_test
    
    def train_single_model(self, model_name: str, X_train: np.ndarray, y_train: np.ndarray,
                          X_test: np.ndarray, y_test: np.ndarray, 
                          optimize_hyperparams: bool = False) -> Dict[str, Any]:
        """Entraîner un seul modèle"""
        
        self.logger.info(f"🤖 Entraînement {model_name}...")
        
        if model_name not in self.available_models:
            raise ValueError(f"Modèle {model_name} non disponible")
        
        # Obtenir modèle
        if optimize_hyperparams:
            model = self._optimize_hyperparameters(model_name, X_train, y_train)
        else:
            model = self.available_models[model_name]
        
        # Normalisation pour modèles linéaires
        if model_name in ['linear', 'ridge', 'lasso']:
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            self.scalers[model_name] = scaler
        else:
            X_train_scaled, X_test_scaled = X_train, X_test
        
        # Entraînement
        start_time = datetime.now()
        model.fit(X_train_scaled, y_train)
        training_time = (datetime.now() - start_time).total_seconds()
        
        # Prédictions
        y_train_pred = model.predict(X_train_scaled)
        y_test_pred = model.predict(X_test_scaled)
        
        # Métriques
        metrics = self._calculate_metrics(y_train, y_train_pred, y_test, y_test_pred)
        metrics.update({
            'model_name': model_name,
            'training_time_seconds': training_time,
            'training_date': datetime.now().isoformat(),
            'hyperparameters_optimized': optimize_hyperparams
        })
        
        # Sauvegarde modèle
        model_info = {
            'model': model,
            'scaler': self.scalers.get(model_name),
            'metrics': metrics,
            'feature_columns': list(X_train.columns) if hasattr(X_train, 'columns') else None
        }
        
        self._save_model(model_info, model_name)
        
        self.logger.info(f"✅ {model_name} - Test MAE: {metrics['test_mae']:.2f}, R²: {metrics['test_r2']:.3f}")
        
        return metrics
    
    def train_ensemble_model(self, X_train: np.ndarray, y_train: np.ndarray,
                           X_test: np.ndarray, y_test: np.ndarray,
                           base_models: List[str] = None) -> Dict[str, Any]:
        """Entraîner modèle d'ensemble"""
        
        if base_models is None:
            base_models = ['xgboost', 'lightgbm', 'random_forest']
        
        self.logger.info(f"🎭 Entraînement ensemble avec {len(base_models)} modèles...")
        
        # Entraîner modèles de base
        base_predictions_train = []
        base_predictions_test = []
        trained_models = []
        
        for model_name in base_models:
            if model_name not in self.available_models:
                continue
                
            model = self.available_models[model_name]
            model.fit(X_train, y_train)
            
            base_predictions_train.append(model.predict(X_train))
            base_predictions_test.append(model.predict(X_test))
            trained_models.append((model_name, model))
        
        # Combiner prédictions (moyenne pondérée)
        weights = self._calculate_ensemble_weights(base_predictions_train, y_train)
        
        y_train_pred = np.average(base_predictions_train, axis=0, weights=weights)
        y_test_pred = np.average(base_predictions_test, axis=0, weights=weights)
        
        # Métriques
        metrics = self._calculate_metrics(y_train, y_train_pred, y_test, y_test_pred)
        metrics.update({
            'model_name': 'ensemble',
            'base_models': base_models,
            'ensemble_weights': weights.tolist(),
            'training_date': datetime.now().isoformat()
        })
        
        # Sauvegarder ensemble
        ensemble_info = {
            'base_models': trained_models,
            'weights': weights,
            'metrics': metrics,
            'feature_columns': list(X_train.columns) if hasattr(X_train, 'columns') else None
        }
        
        self._save_ensemble_model(ensemble_info)
        
        self.logger.info(f"✅ Ensemble - Test MAE: {metrics['test_mae']:.2f}, R²: {metrics['test_r2']:.3f}")
        
        return metrics
    
    def _calculate_ensemble_weights(self, predictions: List[np.ndarray], y_true: np.ndarray) -> np.ndarray:
        """Calculer poids optimaux pour ensemble"""
        errors = [mean_absolute_error(y_true, pred) for pred in predictions]
        
        # Poids inversement proportionnels à l'erreur
        inverse_errors = [1 / (error + 1e-8) for error in errors]
        weights = np.array(inverse_errors) / sum(inverse_errors)
        
        return weights
    
    def _optimize_hyperparameters(self, model_name: str, X_train: np.ndarray, 
                                 y_train: np.ndarray, n_trials: int = 50) -> Any:
        """Optimiser hyperparamètres avec Optuna"""
        
        self.logger.info(f"🔧 Optimisation hyperparamètres {model_name} ({n_trials} essais)...")
        
        def objective(trial):
            if model_name == 'xgboost':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                    'max_depth': trial.suggest_int('max_depth', 4, 12),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                    'random_state': 42
                }
                model = XGBRegressor(**params)
                
            elif model_name == 'lightgbm':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                    'max_depth': trial.suggest_int('max_depth', 4, 12),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                    'feature_fraction': trial.suggest_float('feature_fraction', 0.6, 1.0),
                    'random_state': 42,
                    'verbose': -1
                }
                model = LGBMRegressor(**params)
                
            elif model_name == 'random_forest':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                    'max_depth': trial.suggest_int('max_depth', 5, 20),
                    'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
                    'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 5),
                    'random_state': 42,
                    'n_jobs': -1
                }
                model = RandomForestRegressor(**params)
                
            else:
                return float('inf')
            
            # Validation croisée temporelle
            tscv = TimeSeriesSplit(n_splits=3)
            scores = cross_val_score(model, X_train, y_train, cv=tscv, 
                                   scoring='neg_mean_absolute_error')
            
            return -scores.mean()
        
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        
        # Créer modèle avec meilleurs paramètres
        best_params = study.best_params
        best_params['random_state'] = 42
        
        if model_name == 'xgboost':
            optimized_model = XGBRegressor(**best_params)
        elif model_name == 'lightgbm':
            best_params['verbose'] = -1
            optimized_model = LGBMRegressor(**best_params)
        elif model_name == 'random_forest':
            best_params['n_jobs'] = -1
            optimized_model = RandomForestRegressor(**best_params)
        
        self.logger.info(f"✅ Optimisation terminée - Meilleur score: {study.best_value:.4f}")
        
        return optimized_model
    
    def _calculate_metrics(self, y_train: np.ndarray, y_train_pred: np.ndarray,
                          y_test: np.ndarray, y_test_pred: np.ndarray) -> Dict[str, float]:
        """Calculer métriques de performance complètes"""
        
        metrics = {
            # Métriques train
            'train_mae': mean_absolute_error(y_train, y_train_pred),
            'train_rmse': np.sqrt(mean_squared_error(y_train, y_train_pred)),
            'train_mape': mean_absolute_percentage_error(y_train, y_train_pred) * 100,
            'train_r2': r2_score(y_train, y_train_pred),
            
            # Métriques test
            'test_mae': mean_absolute_error(y_test, y_test_pred),
            'test_rmse': np.sqrt(mean_squared_error(y_test, y_test_pred)),
            'test_mape': mean_absolute_percentage_error(y_test, y_test_pred) * 100,
            'test_r2': r2_score(y_test, y_test_pred),
            
            # Métriques business
            'test_mae_kwh': mean_absolute_error(y_test, y_test_pred),
            'test_mae_fcfa': mean_absolute_error(y_test, y_test_pred) * 150,  # Prix moyen kWh
            'monthly_error_cost_fcfa': mean_absolute_error(y_test, y_test_pred) * 150,
        }
        
        # Éviter overfitting
        metrics['overfitting_ratio'] = metrics['test_mae'] / metrics['train_mae']
        
        return metrics
    
    def _save_model(self, model_info: Dict[str, Any], model_name: str):
        """Sauvegarder modèle et métadonnées"""
        
        # Nom du fichier avec version
        version = self._get_next_version(model_name)
        filename = f"{model_name}_model_v{version}.pkl"
        filepath = self.models_path / filename
        
        # Sauvegarder modèle
        joblib.dump(model_info, filepath)
        
        # Mettre à jour métadonnées
        self._update_model_metadata(model_info['metrics'], model_name, filename)
        
        self.logger.info(f"💾 Modèle sauvegardé: {filepath}")
    
    def _save_ensemble_model(self, ensemble_info: Dict[str, Any]):
        """Sauvegarder modèle d'ensemble"""
        
        version = self._get_next_version('ensemble')
        filename = f"ensemble_model_v{version}.pkl"
        filepath = self.models_path / filename
        
        joblib.dump(ensemble_info, filepath)
        self._update_model_metadata(ensemble_info['metrics'], 'ensemble', filename)
        
        self.logger.info(f"💾 Ensemble sauvegardé: {filepath}")
    
    def _get_next_version(self, model_name: str) -> int:
        """Obtenir prochain numéro de version"""
        import re
        
        pattern = f"{model_name}_model_v(\d+).pkl"
        versions = []
        
        for file in self.models_path.glob(f"{model_name}_model_v*.pkl"):
            match = re.search(pattern, file.name)
            if match:
                versions.append(int(match.group(1)))
        
        return max(versions, default=0) + 1
    
    def _update_model_metadata(self, metrics: Dict[str, Any], model_name: str, filename: str):
        """Mettre à jour métadonnées des modèles"""
        
        metadata_file = self.models_path / 'model_metadata.json'
        
        # Charger métadonnées existantes
        try:
            with open(metadata_file, 'r') as f:
                all_metadata = json.load(f)
        except FileNotFoundError:
            all_metadata = {}
        
        # Ajouter nouvelles métadonnées
        if model_name not in all_metadata:
            all_metadata[model_name] = []
        
        metrics_copy = metrics.copy()
        metrics_copy['filename'] = filename
        all_metadata[model_name].append(metrics_copy)
        
        # Sauvegarder
        with open(metadata_file, 'w') as f:
            json.dump(all_metadata, f, indent=2, default=str)
    
    def train_model(self, data_path: str, model_type: str = 'xgboost', 
                   optimize: bool = False) -> Dict[str, Any]:
        """Interface principale d'entraînement"""
        
        # Charger et préparer données
        df, feature_columns = self.load_and_prepare_data(data_path)
        X_train, X_test, y_train, y_test = self.prepare_train_test_split(df, feature_columns)
        
        # Entraîner selon le type
        if model_type == 'ensemble':
            return self.train_ensemble_model(X_train, y_train, X_test, y_test)
        elif model_type == 'all':
            results = {}
            for model_name in self.available_models.keys():
                try:
                    results[model_name] = self.train_single_model(
                        model_name, X_train, y_train, X_test, y_test, optimize
                    )
                except Exception as e:
                    self.logger.error(f"Erreur {model_name}: {e}")
                    continue
            return results
        else:
            return self.train_single_model(
                model_type, X_train, y_train, X_test, y_test, optimize
            )