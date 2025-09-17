# === FICHIER 9: scripts/train_model.py ===
"""
🚀 Script d'entraînement automatisé des modèles
Utilisation:
    python scripts/train_model.py --model xgboost --optimize
"""

import sys
from pathlib import Path
import argparse
import logging

# Ajouter src au path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from model_trainer import EnergyModelTrainer
from data_processing import DataProcessor


def setup_logging():
    """Configuration du logging"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler("logs/training.log"),
            logging.StreamHandler()
        ]
    )


def main():
    parser = argparse.ArgumentParser(description="Script d'entraînement automatisé")
    parser.add_argument(
        "--model",
        default="xgboost",
        choices=["linear", "xgboost", "lightgbm","random_forest", "ensemble", "all"],
        help="Type de modèle à entraîner"
    )
    parser.add_argument(
        "--data",
        default="data/processed/combined_dataset.csv",
        help="Fichier de données"
    )
    parser.add_argument(
        "--optimize",
        action="store_true",
        help="Optimiser les hyperparamètres"
    )
    parser.add_argument(
        "--prepare-data",
        action="store_true",
        help="Préparer les données avant entraînement"
    )
    args = parser.parse_args()

    setup_logging()
    logger = logging.getLogger(__name__)

    try:
        # Préparer données si demandé
        if args.prepare_data:
            logger.info("🔄 Préparation des données...")
            processor = DataProcessor()
            combined_df, report = processor.process_all_datasets()
            logger.info(f"✅ Données préparées: {len(combined_df)} lignes")
            args.data = "data/processed/combined_dataset.csv"

        # Entraîner modèle(s)
        trainer = EnergyModelTrainer()

        if args.model == "all":
            logger.info("🤖 Entraînement de tous les modèles...")
            models = ["xgboost", "lightgbm", "random_forest"]
            results = {}

            for model in models:
                logger.info(f"📈 Entraînement {model}...")
                result = trainer.train_model(args.data, model, args.optimize)
                results[model] = result

            # Résumé final
            logger.info("📊 RÉSULTATS FINAUX:")
            for model, metrics in results.items():
                if isinstance(metrics, dict):
                    logger.info(
                        f"  {model}: R²={metrics.get('test_r2', 0):.3f}, "
                        f"MAE={metrics.get('test_mae', 0):.2f}"
                    )
        else:
            logger.info(f"🎯 Entraînement {args.model}...")
            result = trainer.train_model(args.data, args.model, args.optimize)

            if isinstance(result, dict):
                logger.info(
                    f"✅ Résultat: R²={result.get('test_r2', 0):.3f}, "
                    f"MAE={result.get('test_mae', 0):.2f}"
                )

        logger.info("🎉 Entraînement terminé avec succès!")

    except Exception as e:
        logger.error(f"❌ Erreur lors de l'entraînement: {e}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
