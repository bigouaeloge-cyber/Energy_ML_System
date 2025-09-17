"""
🔧 Module de traitement et préparation des données
Fonctionnalités:
- Chargement et validation des données brutes
- Nettoyage et harmonisation 
- Fusion des datasets multi-secteurs
- Export vers formats ML
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import yaml

class DataProcessor:
    """Processeur principal pour données énergétiques"""
    
    def __init__(self, config_path: str = 'config/config.yaml'):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.config = self._load_config(config_path)
        
        # Chemins des données
        self.raw_path = Path(self.config['data']['raw_path'])
        self.processed_path = Path(self.config['data']['processed_path'])
        self.external_path = Path(self.config['data']['external_path'])
        
        # Créer dossiers si nécessaire
        for path in [self.raw_path, self.processed_path, self.external_path]:
            path.mkdir(parents=True, exist_ok=True)
    
    def enrich_with_cie_if_available(self, df):
        """Enrichir avec CIE si module disponible"""
        try:
            from cie_ml_integration import CIEMLIntegrator
            integrator = CIEMLIntegrator()
            return integrator.enrich_dataset_with_cie_tarifs(df)
        except ImportError:
            self.logger.warning("Module CIE non disponible, enrichissement ignoré")
            return df
    
    def _load_config(self, config_path: str) -> Dict:
        """Charger configuration YAML"""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            self.logger.warning("Fichier config non trouvé, utilisation config par défaut")
            return self._default_config()
    
    def _default_config(self) -> Dict:
        """Configuration par défaut"""
        return {
            'data': {
                'raw_path': 'data/raw/',
                'processed_path': 'data/processed/',
                'external_path': 'data/external/'
            },
            'business': {
                'default_electricity_price_fcfa': 150
            }
        }
    
    def load_raw_datasets(self) -> Dict[str, pd.DataFrame]:
        """Charger tous les datasets bruts disponibles"""
        self.logger.info("📊 Chargement datasets bruts...")
        
        datasets = {}
        csv_files = list(self.raw_path.glob('*.csv'))
        
        if not csv_files:
            raise FileNotFoundError(f"Aucun fichier CSV trouvé dans {self.raw_path}")
        
        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file)
                dataset_name = csv_file.stem
                datasets[dataset_name] = df
                self.logger.info(f"✅ {dataset_name}: {len(df)} lignes chargées")
            except Exception as e:
                self.logger.error(f"❌ Erreur chargement {csv_file}: {e}")
        
        return datasets
    
    def validate_dataset(self, df: pd.DataFrame, dataset_name: str) -> Dict[str, any]:
        """Valider qualité d'un dataset"""
        validation_results = {
            'dataset_name': dataset_name,
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'missing_values': df.isnull().sum().sum(),
            'duplicate_rows': df.duplicated().sum(),
            'date_range': None,
            'issues': [],
            'is_valid': True
        }
        
        # Vérifier colonnes obligatoires
        required_columns = ['mois', 'kwh_consommes', 'montant_fcfa']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            validation_results['issues'].append(f"Colonnes manquantes: {missing_columns}")
            validation_results['is_valid'] = False
        
        # Vérifier format dates
        if 'mois' in df.columns:
            try:
                date_series = pd.to_datetime(df['mois'])
                validation_results['date_range'] = {
                    'start': date_series.min().strftime('%Y-%m'),
                    'end': date_series.max().strftime('%Y-%m'),
                    'months_count': len(date_series.unique())
                }
                
                # Vérifier continuité temporelle
                if len(date_series.unique()) < len(date_series) * 0.9:  # Au moins 90% de dates uniques
                    validation_results['issues'].append("Dates dupliquées détectées")
                
            except Exception as e:
                validation_results['issues'].append(f"Format date invalide: {e}")
                validation_results['is_valid'] = False
        
        # Vérifier valeurs numériques
        numeric_columns = ['kwh_consommes', 'montant_fcfa']
        for col in numeric_columns:
            if col in df.columns:
                if df[col].dtype not in ['int64', 'float64']:
                    validation_results['issues'].append(f"Colonne {col} non numérique")
                    validation_results['is_valid'] = False
                
                if (df[col] <= 0).any():
                    validation_results['issues'].append(f"Valeurs négatives/nulles dans {col}")
        
        # Détecter outliers
        if 'kwh_consommes' in df.columns:
            q1, q3 = df['kwh_consommes'].quantile([0.25, 0.75])
            iqr = q3 - q1
            outliers = ((df['kwh_consommes'] < q1 - 3*iqr) | 
                       (df['kwh_consommes'] > q3 + 3*iqr)).sum()
            
            if outliers > 0:
                validation_results['issues'].append(f"{outliers} outliers détectés dans consommation")
        
        return validation_results
    
    def clean_dataset(self, df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
        """Nettoyer et standardiser un dataset"""
        self.logger.info(f"🧹 Nettoyage dataset {dataset_name}...")
        
        df_clean = df.copy()
        
        # Standardiser noms colonnes
        column_mapping = {
            'kwh_consommés': 'kwh_consommes',
            'montant_FCFA': 'montant_fcfa',
            'température_moyenne': 'temp_moyenne',
            'nb_jours': 'nb_jours_facture'
        }
        df_clean = df_clean.rename(columns=column_mapping)
        
        # Convertir dates
        if 'mois' in df_clean.columns:
            df_clean['mois'] = pd.to_datetime(df_clean['mois'])
            df_clean = df_clean.sort_values('mois')
        
        # Nettoyer valeurs numériques
        numeric_columns = ['kwh_consommes', 'montant_fcfa', 'temp_moyenne', 'surface_m2']
        for col in numeric_columns:
            if col in df_clean.columns:
                # Convertir en numérique, remplacer erreurs par NaN
                df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
                
                # Supprimer valeurs aberrantes (< 0 ou trop élevées)
                if col == 'kwh_consommes':
                    df_clean = df_clean[(df_clean[col] >= 50) & (df_clean[col] <= 50000)]
                elif col == 'montant_fcfa':
                    df_clean = df_clean[(df_clean[col] >= 7500) & (df_clean[col] <= 7500000)]
                elif col == 'temp_moyenne':
                    df_clean = df_clean[(df_clean[col] >= 15) & (df_clean[col] <= 45)]
        
        # Supprimer doublons
        initial_rows = len(df_clean)
        df_clean = df_clean.drop_duplicates()
        removed_duplicates = initial_rows - len(df_clean)
        
        if removed_duplicates > 0:
            self.logger.info(f"🗑️ {removed_duplicates} doublons supprimés")
        
        # Ajouter métadonnées dataset
        df_clean['source_dataset'] = dataset_name
        df_clean['processing_date'] = datetime.now()
        
        self.logger.info(f"✅ Dataset nettoyé: {len(df_clean)} lignes conservées")
        
        return df_clean
    
    def add_sector_classification(self, df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
        """Ajouter classification sectorielle automatique"""
        df_classified = df.copy()
        
        # Règles de classification basées sur le nom du fichier et les caractéristiques
        if 'maison' in dataset_name.lower() or 'familial' in dataset_name.lower():
            df_classified['secteur'] = 'residential'
            df_classified['type_batiment'] = 'maison'
            
        elif 'bureau' in dataset_name.lower() or 'office' in dataset_name.lower():
            df_classified['secteur'] = 'office'
            df_classified['type_batiment'] = 'bureau'
            
        elif 'boutique' in dataset_name.lower() or 'commerce' in dataset_name.lower():
            df_classified['secteur'] = 'retail'
            df_classified['type_batiment'] = 'commerce'
            
        elif 'hotel' in dataset_name.lower():
            df_classified['secteur'] = 'hotel'
            df_classified['type_batiment'] = 'hotel'
            
        else:
            # Classification automatique basée sur la consommation
            if 'kwh_consommes' in df_classified.columns:
                avg_consumption = df_classified['kwh_consommes'].mean()
                
                if avg_consumption < 1200:  # < 1200 kWh/mois
                    df_classified['secteur'] = 'residential'
                    df_classified['type_batiment'] = 'maison'
                elif avg_consumption < 3000:  # 1200-3000 kWh/mois
                    df_classified['secteur'] = 'office'  
                    df_classified['type_batiment'] = 'bureau'
                else:  # > 3000 kWh/mois
                    df_classified['secteur'] = 'hotel'
                    df_classified['type_batiment'] = 'hotel'
        
        return df_classified
    
    def combine_datasets(self, datasets: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Combiner tous les datasets en un seul"""
        self.logger.info("🔄 Combinaison des datasets...")
        
        processed_datasets = []
        
        for dataset_name, df in datasets.items():
            # Nettoyer dataset
            df_clean = self.clean_dataset(df, dataset_name)
            
            # Ajouter classification sectorielle
            df_classified = self.add_sector_classification(df_clean, dataset_name)
            
            processed_datasets.append(df_classified)
        
        # Combiner tous les datasets
        combined_df = pd.concat(processed_datasets, ignore_index=True, sort=False)
        
        # Harmoniser colonnes communes
        combined_df = self._harmonize_columns(combined_df)
        
        self.logger.info(f"✅ Datasets combinés: {len(combined_df)} lignes totales")
        
        return combined_df
    
    def _harmonize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Harmoniser et créer colonnes communes"""
        df_harmonized = df.copy()
        
        # Colonnes obligatoires avec valeurs par défaut
        required_columns = {
            'nb_personnes': 2,
            'surface_m2': 100,
            'nb_jours_facture': 30,
            'temp_moyenne': 28
        }
        
        for col, default_value in required_columns.items():
            if col not in df_harmonized.columns:
                df_harmonized[col] = default_value
            else:
                df_harmonized[col] = df_harmonized[col].fillna(default_value)
        
        # Calculer métriques dérivées
        df_harmonized['kwh_par_jour'] = df_harmonized['kwh_consommes'] / df_harmonized['nb_jours_facture']
        df_harmonized['kwh_par_m2'] = df_harmonized['kwh_consommes'] / df_harmonized['surface_m2']
        df_harmonized['fcfa_par_kwh'] = df_harmonized['montant_fcfa'] / df_harmonized['kwh_consommes']
        
        return df_harmonized
    
    def process_all_datasets(self) -> Tuple[pd.DataFrame, Dict]:
        """Traiter tous les datasets et générer rapport"""
        # Charger datasets bruts
        raw_datasets = self.load_raw_datasets()
        
        # Valider chaque dataset
        validation_reports = {}
        for name, df in raw_datasets.items():
            validation_reports[name] = self.validate_dataset(df, name)
        
        # Combiner datasets valides
        valid_datasets = {
            name: df for name, df in raw_datasets.items()
            if validation_reports[name]['is_valid']
        }
        
        if not valid_datasets:
            raise ValueError("Aucun dataset valide trouvé")
        
        combined_df = self.combine_datasets(valid_datasets)
        
        # Sauvegarder dataset combiné
        output_path = self.processed_path / 'combined_dataset.csv'
        combined_df.to_csv(output_path, index=False)
        self.logger.info(f"💾 Dataset combiné sauvegardé: {output_path}")
        
        # Générer rapport de traitement
        processing_report = self._generate_processing_report(
            raw_datasets, combined_df, validation_reports
        )
        
        return combined_df, processing_report
    
    def _generate_processing_report(self, raw_datasets: Dict, 
                                  combined_df: pd.DataFrame, 
                                  validation_reports: Dict) -> Dict:
        """Générer rapport détaillé du traitement"""
        report = {
            'processing_date': datetime.now().isoformat(),
            'datasets_processed': len(raw_datasets),
            'total_raw_records': sum(len(df) for df in raw_datasets.values()),
            'final_records': len(combined_df),
            'data_retention_rate': len(combined_df) / sum(len(df) for df in raw_datasets.values()),
            'sectors_detected': combined_df['secteur'].value_counts().to_dict(),
            'date_range': {
                'start': combined_df['mois'].min().strftime('%Y-%m'),
                'end': combined_df['mois'].max().strftime('%Y-%m'),
                'total_months': len(combined_df['mois'].unique())
            },
            'validation_summary': validation_reports,
            'data_quality': {
                'missing_values': combined_df.isnull().sum().to_dict(),
                'consumption_stats': {
                    'mean_kwh': combined_df['kwh_consommes'].mean(),
                    'std_kwh': combined_df['kwh_consommes'].std(),
                    'min_kwh': combined_df['kwh_consommes'].min(),
                    'max_kwh': combined_df['kwh_consommes'].max()
                }
            }
        }
        
        # Sauvegarder rapport
        report_path = self.processed_path / 'processing_report.json'
        import json
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        return report
    
    def export_for_ml(self, df: pd.DataFrame, target_format: str = 'ml_ready') -> pd.DataFrame:
        """Exporter données formatées pour ML"""
        
        if target_format == 'ml_ready':
            return self._prepare_ml_features(df)
        elif target_format == 'time_series':
            return self._prepare_time_series(df)
        else:
            raise ValueError(f"Format {target_format} non supporté")
    
    def _prepare_ml_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Préparer features pour modèles ML classiques"""
        from feature_engineering import AdvancedFeatureEngineer
        
        feature_engineer = AdvancedFeatureEngineer()
        df_features = feature_engineer.create_comprehensive_features(df)
        
        return df_features
    
    def _prepare_time_series(self, df: pd.DataFrame) -> pd.DataFrame:
        """Préparer données pour modèles de séries temporelles"""
        df_ts = df.copy()
        
        # Assurer index temporel
        df_ts = df_ts.set_index('mois').sort_index()
        
        # Interpoler valeurs manquantes
        numeric_cols = df_ts.select_dtypes(include=[np.number]).columns
        df_ts[numeric_cols] = df_ts[numeric_cols].interpolate(method='time')
        
        return df_ts