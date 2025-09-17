import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Ajouter src au path pour que Python trouve ton module
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from data_processing import DataProcessor
@pytest.fixture
def processor(self):
    """Fixture pour créer un processeur de données"""
    return DataProcessor()

@pytest.fixture 
def sample_dataset(self):
    """Fixture pour créer un dataset de test"""
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='M')
    
    data = []
    for i, date in enumerate(dates):
        # Simuler données réalistes
        base_consumption = 1000 + np.random.normal(0, 100)
        seasonal_variation = 200 * np.sin(2 * np.pi * date.month / 12)
        consumption = base_consumption + seasonal_variation
        
        data.append({
            'mois': date.strftime('%Y-%m'),
            'kwh_consommes': max(400, consumption),
            'montant_fcfa': max(400, consumption) * 150,
            'nb_jours_facture': 30,
            'surface_m2': 150,
            'temp_moyenne': 28 + 3 * np.sin(2 * np.pi * date.month / 12)
        })
    
    return pd.DataFrame(data)

def test_processor_initialization(self, processor):
    """Test initialisation du processeur"""
    assert processor is not None
    assert hasattr(processor, 'raw_path')
    assert hasattr(processor, 'processed_path')
    assert hasattr(processor, 'external_path')

def test_validate_dataset_valid(self, processor, sample_dataset):
    """Test validation d'un dataset valide"""
    validation = processor.validate_dataset(sample_dataset, 'test_dataset')
    
    assert validation['is_valid'] is True
    assert validation['total_rows'] == len(sample_dataset)
    assert validation['dataset_name'] == 'test_dataset'
    assert len(validation['issues']) == 0

def test_validate_dataset_missing_columns(self, processor):
    """Test validation avec colonnes manquantes"""
    invalid_df = pd.DataFrame({
        'mois': ['2023-01', '2023-02'],
        'other_col': [1, 2]
        # kwh_consommes et montant_fcfa manquantes
    })
    
    validation = processor.validate_dataset(invalid_df, 'invalid_dataset')
    
    assert validation['is_valid'] is False
    assert any('Colonnes manquantes' in issue for issue in validation['issues'])

def test_clean_dataset(self, processor, sample_dataset):
    """Test nettoyage d'un dataset"""
    # Ajouter quelques données problématiques
    dirty_df = sample_dataset.copy()
    dirty_df.loc[0, 'kwh_consommes'] = -100  # Valeur négative
    dirty_df.loc[1, 'temp_moyenne'] = 100    # Valeur aberrante
    dirty_df = pd.concat([dirty_df, dirty_df.iloc[0:1]])  # Doublon
    
    cleaned_df = processor.clean_dataset(dirty_df, 'test')
    
    # Vérifier que les problèmes ont été corrigés
    assert len(cleaned_df) < len(dirty_df)  # Doublons supprimés
    assert all(cleaned_df['kwh_consommes'] >= 50)  # Valeurs négatives supprimées
    assert all(cleaned_df['temp_moyenne'] <= 45)   # Aberrations supprimées
    assert 'source_dataset' in cleaned_df.columns

def test_add_sector_classification(self, processor, sample_dataset):
    """Test classification sectorielle"""
    # Test avec différents noms de dataset
    residential = processor.add_sector_classification(sample_dataset, 'maison_familiale')
    office = processor.add_sector_classification(sample_dataset, 'bureau_pme')
    hotel = processor.add_sector_classification(sample_dataset, 'petit_hotel')
    
    assert all(residential['secteur'] == 'residential')
    assert all(office['secteur'] == 'office') 
    assert all(hotel['secteur'] == 'hotel')
    
    # Test classification automatique basée sur consommation
    high_consumption_df = sample_dataset.copy()
    high_consumption_df['kwh_consommes'] = 4000  # Haute consommation
    
    classified = processor.add_sector_classification(high_consumption_df, 'unknown')
    assert all(classified['secteur'] == 'hotel')  # Doit classifier comme hôtel

def test_harmonize_columns(self, processor):
    """Test harmonisation des colonnes"""
    # Dataset avec colonnes manquantes
    incomplete_df = pd.DataFrame({
        'mois': ['2023-01', '2023-02'],
        'kwh_consommes': [1000, 1100],
        'montant_fcfa': [150000, 165000]
        # surface_m2, nb_personnes manquantes
    })
    
    harmonized = processor._harmonize_columns(incomplete_df)
    
    # Vérifier que les colonnes manquantes ont été ajoutées avec valeurs par défaut
    required_cols = ['nb_personnes', 'surface_m2', 'nb_jours_facture', 'temp_moyenne']
    for col in required_cols:
        assert col in harmonized.columns
        assert not harmonized[col].isna().any()
    
    # Vérifier calcul métriques dérivées
    assert 'kwh_par_jour' in harmonized.columns
    assert 'kwh_par_m2' in harmonized.columns
    assert 'fcfa_par_kwh' in harmonized.columns

def test_combine_datasets(self, processor, sample_dataset):
    """Test combinaison de plusieurs datasets"""
    # Créer plusieurs datasets avec différents secteurs
    dataset1 = sample_dataset.copy()
    dataset2 = sample_dataset.copy()
    dataset2['kwh_consommes'] *= 1.5  # Consommation différente
    
    datasets = {
        'maison': dataset1,
        'bureau': dataset2
    }
    
    combined = processor.combine_datasets(datasets)
    
    # Vérifications
    assert len(combined) == len(dataset1) + len(dataset2)
    assert 'secteur' in combined.columns
    assert set(combined['secteur'].unique()) == {'residential', 'office'}
    assert 'source_dataset' in combined.columns

def test_export_for_ml(self, processor, sample_dataset):
    """Test export pour ML"""
    # Test format ML ready
    ml_ready = processor.export_for_ml(sample_dataset, 'ml_ready')
    
    # Vérifier que des features avancées ont été ajoutées
    expected_features = ['mois_sin', 'mois_cos', 'kwh_lag_1', 'kwh_ma_3']
    for feature in expected_features:
        assert feature in ml_ready.columns
    
    # Test format time series
    ts_ready = processor.export_for_ml(sample_dataset, 'time_series')
    
    # Vérifier que l'index est temporel
    assert isinstance(ts_ready.index, pd.DatetimeIndex)

@pytest.mark.integration
def test_process_all_datasets_integration(self, processor):
    """Test d'intégration complet (nécessite des fichiers de données)"""
    # Ce test nécessite des vrais fichiers CSV dans data/raw/
    # Ignorer si pas de fichiers disponibles
    
    raw_files = list(processor.raw_path.glob('*.csv'))
    if not raw_files:
        pytest.skip("Aucun fichier de données disponible pour test intégration")
    
    try:
        combined_df, report = processor.process_all_datasets()
        
        # Vérifications du résultat
        assert isinstance(combined_df, pd.DataFrame)
        assert len(combined_df) > 0
        assert isinstance(report, dict)
        assert 'processing_date' in report
        assert 'final_records' in report
        
    except Exception as e:
        pytest.fail(f"Erreur traitement intégration: {e}")