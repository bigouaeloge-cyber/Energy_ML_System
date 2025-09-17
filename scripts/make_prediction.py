#!/usr/bin/env python3
"""
Script utilitaire pour générer des prédictions indépendamment
de l'application principale.
"""

import sys
from pathlib import Path
import logging

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from predictor import EnergyPredictor

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def main():
    logger = logging.getLogger(__name__)
    
    # Exemple de prédiction pour un bureau
    sector = 'office'
    params = {
        'surface_m2': 200,
        'nb_personnes': 10
    }
    horizon = '1month'
    
    predictor = EnergyPredictor()
    results = predictor.predict_sector(sector, params, horizon)
    
    logger.info("Prédictions générées :")
    logger.info(results)
    
    # Sauvegarde en CSV
    predictor.save_predictions(results, f'predictions_{sector}.csv')
    logger.info(f"✅ Prédictions sauvegardées dans predictions_{sector}.csv")

if __name__ == '__main__':
    main()
