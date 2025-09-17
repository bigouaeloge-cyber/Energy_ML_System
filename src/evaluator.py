import logging
import pandas as pd

class ModelEvaluator:
    """Classe pour évaluer les modèles ML"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def evaluate_single_model(self, model_name):
        self.logger.info(f"Évaluation du modèle {model_name}...")
        # Ici tu peux charger le modèle et le jeu de test
        # Simulation d'un résultat
        results = pd.DataFrame({
            'metric': ['MAE', 'RMSE', 'R2'],
            'value': [120.5, 150.7, 0.87]
        })
        return results
    
    def evaluate_all_models(self):
        self.logger.info("Évaluation de tous les modèles...")
        models = ['xgboost', 'lightgbm', 'ensemble']
        all_results = {}
        for model in models:
            all_results[model] = self.evaluate_single_model(model)
        return all_results
    
    def display_results(self, results, metrics_type='detailed'):
        self.logger.info("Résultats de l'évaluation :")
        if isinstance(results, dict):
            for model, df in results.items():
                self.logger.info(f"\n--- {model} ---")
                self.logger.info(df.to_string(index=False))
        else:
            self.logger.info(results.to_string(index=False))
    
    def generate_report(self, results):
        # Générer un rapport PDF ou Excel (simulation ici)
        self.logger.info("Génération du rapport d'évaluation (simulé)...")
        # Exemple: enregistrer en CSV
        if isinstance(results, dict):
            for model, df in results.items():
                df.to_csv(f'reports/{model}_evaluation.csv', index=False)
        else:
            results.to_csv('reports/evaluation.csv', index=False)
        self.logger.info("✅ Rapport généré dans le dossier reports/")
