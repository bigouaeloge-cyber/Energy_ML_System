#!/usr/bin/env python3
"""
🚀 Energy ML System - Point d'entrée principal
Système d'optimisation énergétique par Intelligence Artificielle
Développé pour le marché PME Côte d'Ivoire

Usage:
    python main.py train --model xgboost
    python main.py predict --sector hotel
    python main.py dashboard
    python main.py evaluate --all
    python main.py deploy --target raspberry_pi
"""

import argparse
import sys
sys.stdout.reconfigure(encoding='utf-8')
import os
os.environ['PYTHONIOENCODING'] = 'utf-8'
from pathlib import Path
import logging
from datetime import datetime

# Ajouter src au Python path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

# Imports des modules personnalisés
from model_trainer import EnergyModelTrainer
from predictor import EnergyPredictor
from evaluator import ModelEvaluator
from data_processing import DataProcessor
from weather_api import WeatherEnrichment
from cie_tarification import CIETarificationSystem
from cie_ml_integration import CIEMLIntegrator, CIEClientAdvisor
import pandas as pd

# Configuration logging global
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/system.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)

def setup_directories():
    """Créer dossiers nécessaires s'ils n'existent pas"""
    directories = [
        'data/raw', 'data/processed', 'data/external',
        'models', 'logs', 'notebooks', 'config'
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)

def main():
    """Fonction principale du système"""
    setup_directories()
    
    parser = argparse.ArgumentParser(
        description='🚀 Energy ML System - AI pour optimisation énergétique',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples d'utilisation:
  python main.py train --model xgboost --data all
  python main.py predict --sector hotel --params surface:200,chambres:15
  python main.py dashboard --port 8501
  python main.py evaluate --model all --report detailed
  python main.py deploy --target raspberry_pi --config production
        """
    )
    
    # Sous-commandes principales
    subparsers = parser.add_subparsers(dest='command', help='Commandes disponibles')
    
    # === COMMANDE TRAIN ===
    train_parser = subparsers.add_parser('train', help='Entraîner modèles ML')
    train_parser.add_argument('--model', choices=['linear', 'xgboost', 'lightgbm', 'random_forest', 'ensemble', 'all'], 
                             default='xgboost', help='Type de modèle')
    train_parser.add_argument('--data', default='data/processed/combined_dataset.csv',
                             help='Fichier de données')
    train_parser.add_argument('--optimize', action='store_true', 
                             help='Optimisation hyperparamètres')
    train_parser.add_argument('--cv-folds', type=int, default=5,
                             help='Nombre de folds pour validation croisée')
    
    # === COMMANDE PREDICT ===
    predict_parser = subparsers.add_parser('predict', help='Faire des prédictions')
    predict_parser.add_argument('--sector', choices=['residential', 'office', 'retail', 'hotel'],
                               required=True, help='Secteur cible')
    predict_parser.add_argument('--model', default='best', help='Modèle à utiliser')
    predict_parser.add_argument('--params', help='Paramètres (ex: surface:200,chambres:15)')
    predict_parser.add_argument('--horizon', choices=['1h', '24h', '1month', '1year'],
                               default='1month', help='Horizon de prédiction')
    predict_parser.add_argument('--output', help='Fichier de sortie (CSV/JSON)')
    
    # === COMMANDE DASHBOARD ===
    dashboard_parser = subparsers.add_parser('dashboard', help='Lancer interface web')
    dashboard_parser.add_argument('--port', type=int, default=8501, help='Port web')
    dashboard_parser.add_argument('--host', default='localhost', help='Adresse host')
    dashboard_parser.add_argument('--theme', choices=['light', 'dark'], default='light')
    
    # === COMMANDE EVALUATE ===
    evaluate_parser = subparsers.add_parser('evaluate', help='Évaluer performances')
    evaluate_parser.add_argument('--model', default='all', help='Modèle(s) à évaluer')
    evaluate_parser.add_argument('--metrics', choices=['basic', 'detailed', 'business'], 
                                default='detailed', help='Type de métriques')
    evaluate_parser.add_argument('--report', action='store_true', help='Générer rapport PDF')
    
    # === COMMANDE DEPLOY ===
    deploy_parser = subparsers.add_parser('deploy', help='Déployer système')
    deploy_parser.add_argument('--target', choices=['raspberry_pi', 'cloud', 'local'], 
                              required=True, help='Cible de déploiement')
    deploy_parser.add_argument('--config', choices=['dev', 'staging', 'production'], 
                              default='dev', help='Configuration')
    
    # === COMMANDES UTILITAIRES ===
    subparsers.add_parser('setup', help='Configuration initiale du système')
    subparsers.add_parser('status', help='Status du système')
    subparsers.add_parser('clean', help='Nettoyer fichiers temporaires')
    
    args = parser.parse_args()
    
    # Logger principal
    logger = logging.getLogger(__name__)
    
    try:
        if args.command == 'train':
            logger.info("🤖 Démarrage entraînement modèle(s)...")
            trainer = EnergyModelTrainer()
            
            if args.model == 'all':
                models = ['xgboost', 'lightgbm', 'random_forest' 'ensemble']
                for model in models:
                    trainer.train_model(args.data, model, optimize=args.optimize)
            else:
                trainer.train_model(args.data, args.model, optimize=args.optimize)
        
        elif args.command == 'predict':
            logger.info("🔮 Génération prédictions...")
            predictor = EnergyPredictor()
            
            # Parser paramètres
            params = {}
            if args.params:
                for param in args.params.split(','):
                    key, value = param.split(':')
                    params[key.strip()] = float(value.strip()) if value.strip().isdigit() else value.strip()
            
            # Faire prédictions
            results = predictor.predict_sector(args.sector, params, args.horizon)
            
            if args.output:
                predictor.save_predictions(results, args.output)
            else:
                predictor.display_predictions(results)
            
            # --- CALCULS CIE ET OPTIMISATION CLIENT ---
            integrator = CIEMLIntegrator()
            df_results = pd.DataFrame(results) if not isinstance(results, pd.DataFrame) else results
            df_enriched = integrator.enrich_dataset_with_cie_tarifs(df_results)

            advisor = CIEClientAdvisor()
            client_analysis = advisor.analyze_client_bill(
                kwh_monthly=1200,
                current_bill_fcfa=180000,
                sector='office',
                surface_m2=200
            )
            logger.info(f"Économies potentielles: {client_analysis['recommandations']['economies_annuelles_potentielles']:,} FCFA/an")

            cie = CIETarificationSystem()
            bill = cie.calculate_bill(1200, "professionnel_bt", 11.0, "simple")
            logger.info(f"Facture calculée: {bill['total_ttc']:,} FCFA")

            optimal = cie.find_optimal_tariff(1800, 11.0, "professionnel")
            logger.info(f"Tarif optimal: {optimal['optimal']['tarif']}")

        elif args.command == 'dashboard':
            logger.info(f"📊 Lancement dashboard sur {args.host}:{args.port}")
            os.system(f'streamlit run app/dashboard.py --server.port {args.port} --server.address {args.host}')
        
        elif args.command == 'evaluate':
            logger.info("📈 Évaluation des modèles...")
            evaluator = ModelEvaluator()
            
            if args.model == 'all':
                results = evaluator.evaluate_all_models()
            else:
                results = evaluator.evaluate_single_model(args.model)
            
            evaluator.display_results(results, args.metrics)
            
            if args.report:
                evaluator.generate_report(results)
        
        elif args.command == 'deploy':
            logger.info(f"🚀 Déploiement sur {args.target}...")
            from deployment import SystemDeployer
            deployer = SystemDeployer(args.target, args.config)
            deployer.deploy()
        
        elif args.command == 'setup':
            logger.info("⚙️ Configuration initiale...")
            setup_system()
        
        elif args.command == 'status':
            logger.info("📊 Vérification status système...")
            check_system_status()
        
        elif args.command == 'clean':
            logger.info("🧹 Nettoyage fichiers temporaires...")
            clean_temp_files()
        
        else:
            parser.print_help()
            
    except KeyboardInterrupt:
        logger.info("⏹️ Interruption utilisateur")
        return 130
    except Exception as e:
        logger.error(f"❌ Erreur critique: {e}", exc_info=True)
        return 1
    
    logger.info("✅ Opération terminée avec succès")
    return 0


def setup_system():
    """Configuration initiale du système"""
    logger = logging.getLogger(__name__)
    
    # Vérifier dépendances
    try:
        import pandas, numpy, sklearn, xgboost, streamlit
        logger.info("✅ Dépendances Python OK")
    except ImportError as e:
        logger.error(f"❌ Dépendance manquante: {e}")
        logger.info("Exécutez: pip install -r requirements.txt")
        return
    
    # Créer fichiers de configuration
    create_config_files()
    
    # Télécharger données d'exemple si nécessaire
    create_sample_data()
    
    logger.info("🎉 Configuration terminée!")

def create_config_files():
    """Créer fichiers de configuration par défaut"""
    
    # config.yaml
    config_content = """
# Configuration systeme Energy ML
system:
  name: "Energy ML System"
  version: "1.0.0"
  environment: "development"

data:
  raw_path: "data/raw/"
  processed_path: "data/processed/"
  external_path: "data/external/"

models:
  available:
    - "xgboost"
    - "lightgbm"
    - "random_forest"
  default: "xgboost"
  model_path: "models/"
  auto_retrain: true
  performance_threshold: 0.80


api:
  openweather_base_url: "http://api.openweathermap.org/data/2.5"
  location:
    city: "Abidjan"
    country: "CI"
    latitude: 5.3600
    longitude: -4.0083

dashboard:
  title: "Interface de Données Energétiques"
  theme: "light"
  refresh_interval: 300  # seconds

business:
  default_electricity_price_fcfa: 150
  target_savings_percent: 25
  roi_target_months: 12

logging:
  level: "INFO"
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
  file: "logs/system.log"
"""
    
    with open('config/config.yaml', 'w', encoding='utf-8') as f:
        f.write(config_content.strip())
    
    # .env template
    env_content = """
# Clés API (GARDEZ SECRETES !)
OPENWEATHER_API_KEY=your_api_key_here
WANDB_API_KEY=your_wandb_key_here

# Base de données (optionnel)
DATABASE_URL=sqlite:///energy_ml.db

# Configuration production
FLASK_SECRET_KEY=your_secret_key_here
STREAMLIT_SERVER_PORT=8501

# Alertes (optionnel)
TWILIO_ACCOUNT_SID=your_twilio_sid
TWILIO_AUTH_TOKEN=your_twilio_token
WHATSAPP_NUMBER=+225XXXXXXXX
"""
    
    with open('config/api_keys.env', 'w', encoding='utf-8') as f:
        f.write(env_content.strip())

def create_sample_data():
    """Créer données d'exemple pour tests"""
    import pandas as pd
    import numpy as np
    from datetime import datetime, timedelta
    
    # Données exemple maison familiale
    dates = pd.date_range(start='2023-01-01', end='2024-12-31', freq='M')
    
    sample_data = []
    for i, date in enumerate(dates):
        # Simulation consommation réaliste
        base_consumption = 800  # kWh base
        seasonal_variation = 200 * np.sin(2 * np.pi * date.month / 12)  # Variation saisonnière
        random_noise = np.random.normal(0, 50)
        
        consumption = base_consumption + seasonal_variation + random_noise
        
        sample_data.append({
            'mois': date.strftime('%Y-%m'),
            'kwh_consommes': max(400, consumption),  # Min 400 kWh
            'montant_fcfa': max(400, consumption) * 150,
            'nb_jours_facture': 30,
            'type_logement': 'maison',
            'nb_personnes': 4,
            'surface_m2': 150,
            'temp_moyenne': 28 + 3 * np.sin(2 * np.pi * date.month / 12)
        })
    
    df = pd.DataFrame(sample_data)
    df.to_csv('data/raw/maison_familiale.csv', index=False)
    
    print("✅ Données d'exemple créées: data/raw/maison_familiale.csv")

def check_system_status():
    """Vérifier status général du systeme"""
    logger = logging.getLogger(__name__)
    
    status = {
        'models': check_models_status(),
        'data': check_data_status(),
        'dependencies': check_dependencies_status()
    }
    
    # Afficher status
    for category, items in status.items():
        logger.info(f"\n=== {category.upper()} ===")
        for item, is_ok in items.items():
            status_emoji = "✅" if is_ok else "❌"
            logger.info(f"{status_emoji} {item}")

def check_models_status():
    """Vérifier status des modèles"""
    models_dir = Path('models')
    if not models_dir.exists():
        return {"Dossier models": False}
    
    model_files = list(models_dir.glob('*.pkl'))
    return {
        "Dossier models": True,
        f"Modèles disponibles ({len(model_files)})": len(model_files) > 0,
        "Métadonnées": (models_dir / 'model_metadata.json').exists()
    }

def check_data_status():
    """Vérifier status des données"""
    data_dir = Path('data')
    return {
        "Dossier data": data_dir.exists(),
        "Données raw": len(list((data_dir / 'raw').glob('*.csv'))) > 0 if data_dir.exists() else False,
        "Données processed": (data_dir / 'processed').exists() if data_dir.exists() else False
    }

def check_dependencies_status():
    """Vérifier dépendances Python"""
    dependencies = ['pandas', 'numpy', 'sklearn', 'xgboost', 'streamlit']
    status = {}
    
    for dep in dependencies:
        try:
            __import__(dep)
            status[dep] = True
        except ImportError:
            status[dep] = False
    
    return status

def clean_temp_files():
    """Nettoyer fichiers temporaires"""
    import shutil
    
    temp_patterns = [
        'logs/*.log.*',  # Anciens logs
        'data/external/.cache',  # Cache météo
        '**/__pycache__',  # Cache Python
        '**/*.pyc',  # Fichiers Python compilés
        '.pytest_cache'  # Cache tests
    ]
    
    cleaned = 0
    for pattern in temp_patterns:
        for file_path in Path('.').glob(pattern):
            if file_path.is_file():
                file_path.unlink()
                cleaned += 1
            elif file_path.is_dir():
                shutil.rmtree(file_path)
                cleaned += 1
    
    print(f"🧹 {cleaned} fichiers temporaires supprimés")

if __name__ == '__main__':
    exit(main())