"""
🔗 Intégration du module CIE dans le système Energy ML
Enrichissement automatique des données avec tarification précise
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List, Optional

# Import du module CIE
from cie_tarification import CIETarificationSystem, calculate_switching_savings

class CIEMLIntegrator:
    """Intégrateur CIE pour le système ML"""
    
    def __init__(self):
        self.cie_system = CIETarificationSystem()
        
    def enrich_dataset_with_cie_tarifs(self, df: pd.DataFrame) -> pd.DataFrame:
        """Enrichir dataset avec tarification CIE précise"""
        
        df_enriched = df.copy()
        
        # Ajouter colonnes CIE
        cie_columns = [
            'tarif_cie_optimal',
            'option_cie_optimale', 
            'cout_reel_calcule',
            'prix_kwh_reel_calcule',
            'economies_potentielles_cie',
            'efficacite_tarifaire',
            'tranche_recommandee',
            'puissance_optimale_kva'
        ]
        
        for col in cie_columns:
            df_enriched[col] = np.nan
        
        # Traiter chaque ligne
        for idx, row in df_enriched.iterrows():
            try:
                # Déterminer type de tarif actuel (approximation basée sur consommation/type)
                tarif_actuel = self._detect_current_tariff(row)
                puissance_kva = self._estimate_puissance_kva(row)
                
                # Calculer coût réel avec CIE
                bill_details = self.cie_system.calculate_bill(
                    row['kwh_consommes'],
                    tarif_actuel,
                    puissance_kva
                )
                
                # Trouver tarif optimal
                usage_type = self._determine_usage_type(row)
                optimal_tariff = self.cie_system.find_optimal_tariff(
                    row['kwh_consommes'],
                    puissance_kva,
                    usage_type
                )
                
                # Remplir les nouvelles colonnes
                df_enriched.loc[idx, 'cout_reel_calcule'] = bill_details['total_ttc']
                df_enriched.loc[idx, 'prix_kwh_reel_calcule'] = bill_details['prix_moyen_kwh']
                df_enriched.loc[idx, 'efficacite_tarifaire'] = row['montant_fcfa'] / bill_details['total_ttc']
                
                if 'optimal' in optimal_tariff:
                    optimal = optimal_tariff['optimal']
                    df_enriched.loc[idx, 'tarif_cie_optimal'] = optimal['tarif']
                    df_enriched.loc[idx, 'option_cie_optimale'] = optimal['option']
                    
                    # Calcul économies potentielles
                    current_cost = row['montant_fcfa']
                    optimal_cost = optimal['cout_total']
                    economies = max(0, current_cost - optimal_cost)
                    df_enriched.loc[idx, 'economies_potentielles_cie'] = economies
                
                df_enriched.loc[idx, 'puissance_optimale_kva'] = puissance_kva
                
            except Exception as e:
                print(f"Erreur traitement ligne {idx}: {e}")
                continue
        
        return df_enriched
    
    def _detect_current_tariff(self, row: pd.Series) -> str:
        """Détecter le tarif CIE actuel basé sur les données"""
        
        kwh = row['kwh_consommes']
        montant = row['montant_fcfa']
        prix_moyen = montant / kwh if kwh > 0 else 0
        
        # Logique de détection basée sur prix moyen et type
        if hasattr(row, 'secteur'):
            secteur = row.get('secteur', 'residential')
        else:
            # Approximation basée sur consommation
            if kwh < 500:
                secteur = 'residential'
            elif kwh < 3000:
                secteur = 'office'
            else:
                secteur = 'hotel'
        
        if secteur == 'residential':
            if prix_moyen < 100:
                return 'domestique_social'
            else:
                return 'domestique_standard'
        elif secteur in ['office', 'retail']:
            if kwh > 2000:
                return 'professionnel_mt'
            else:
                return 'professionnel_bt'
        elif secteur == 'hotel':
            if kwh > 5000:
                return 'professionnel_mt'
            else:
                return 'professionnel_bt'
        else:
            return 'domestique_standard'
    
    def _estimate_puissance_kva(self, row: pd.Series) -> float:
        """Estimer puissance souscrite basée sur consommation et type"""
        
        kwh = row['kwh_consommes']
        surface = row.get('surface_m2', 100)
        
        # Estimation basée sur consommation mensuelle
        # Approximation: puissance = consommation_pointe / (0.7 * heures_utilisation)
        
        if kwh < 300:
            return 3.3  # 15A
        elif kwh < 600:
            return 5.5  # 25A
        elif kwh < 1200:
            return 11.0  # 50A
        elif kwh < 2500:
            return 22.0  # 100A
        elif kwh < 5000:
            return 36.0  # 160A
        else:
            # Pour gros consommateurs, estimer selon surface
            return min(250, max(36, surface / 10))
    
    def _determine_usage_type(self, row: pd.Series) -> str:
        """Déterminer type d'usage pour optimisation tarifaire"""
        
        if hasattr(row, 'secteur'):
            secteur = row.get('secteur', 'residential')
            if secteur == 'residential':
                return 'domestique'
            elif secteur in ['office', 'retail', 'hotel']:
                return 'professionnel'
            else:
                return 'industriel'
        else:
            # Approximation basée sur consommation
            kwh = row['kwh_consommes']
            if kwh < 1000:
                return 'domestique'
            elif kwh < 8000:
                return 'professionnel'
            else:
                return 'industriel'
    
    def create_cie_optimization_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Créer features spécialisés optimisation CIE"""
        
        df_features = df.copy()
        
        # Variables d'efficacité tarifaire
        df_features['sur_facturation_ratio'] = df_features['montant_fcfa'] / df_features['cout_reel_calcule']
        df_features['potentiel_economie_pct'] = (df_features['economies_potentielles_cie'] / df_features['montant_fcfa'] * 100).fillna(0)
        
        # Indicateurs de mauvais dimensionnement tarifaire
        df_features['tarif_sous_optimal'] = (df_features['economies_potentielles_cie'] > df_features['montant_fcfa'] * 0.1).astype(int)
        df_features['changement_tranche_profitable'] = (df_features['economies_potentielles_cie'] > 10000).astype(int)  # >10k FCFA économies
        
        # Variables de seuils CIE
        df_features['proche_seuil_tranche'] = df_features['kwh_consommes'].apply(self._detect_threshold_proximity)
        df_features['optimisation_puissance_possible'] = df_features.apply(self._detect_power_optimization, axis=1)
        
        # Simulation HP/HC
        df_features['economie_hp_hc_estimee'] = df_features.apply(self._estimate_hp_hc_savings, axis=1)
        
        return df_features
    
    def _detect_threshold_proximity(self, kwh: float) -> int:
        """Détecter proximité des seuils de tranches tarifaires"""
        
        seuils_domestique = [110, 400]  # Seuils tranches domestique
        seuils_pro = [2000, 5000]  # Seuils approximatifs changement tarif
        
        tous_seuils = seuils_domestique + seuils_pro
        
        for seuil in tous_seuils:
            if abs(kwh - seuil) / seuil < 0.1:  # Dans les 10% du seuil
                return 1
        
        return 0
    
    def _detect_power_optimization(self, row: pd.Series) -> int:
        """Détecter si optimisation puissance possible"""
        
        kwh = row['kwh_consommes']
        puissance_actuelle = row.get('puissance_optimale_kva', 11.0)
        
        # Estimation puissance réellement nécessaire
        puissance_necessaire = self._estimate_puissance_kva(row)
        
        # Si puissance souscrite > 150% de nécessaire → optimisation possible
        if puissance_actuelle > puissance_necessaire * 1.5:
            return 1
        
        return 0
    
    def _estimate_hp_hc_savings(self, row: pd.Series) -> float:
        """Estimer économies potentielles tarif HP/HC"""
        
        kwh = row['kwh_consommes']
        tarif_actuel = row.get('tarif_cie_optimal', 'professionnel_bt')
        puissance = row.get('puissance_optimale_kva', 11.0)
        
        if 'professionnel' not in tarif_actuel or kwh < 1000:
            return 0
        
        try:
            # Calculer coût tarif simple
            cout_simple = self.cie_system.calculate_bill(kwh, tarif_actuel, puissance, 'simple')
            
            # Calculer coût HP/HC
            cout_hp_hc = self.cie_system.calculate_bill(kwh, tarif_actuel, puissance, 'heures_pleines_creuses')
            
            economie = cout_simple['total_ttc'] - cout_hp_hc['total_ttc']
            return max(0, economie)
            
        except:
            return 0
    
    def generate_cie_recommendations(self, row: pd.Series) -> Dict:
        """Générer recommandations CIE personnalisées"""
        
        recommendations = {
            'tarif_actuel_detecte': row.get('tarif_cie_optimal', 'unknown'),
            'actions_recommandees': [],
            'economies_annuelles_potentielles': 0,
            'priorite': 'basse'
        }
        
        # Recommandation changement de tarif
        economies_cie = row.get('economies_potentielles_cie', 0)
        if economies_cie > 5000:  # >5k FCFA/mois
            recommendations['actions_recommandees'].append({
                'action': 'Changement de tranche tarifaire',
                'nouveau_tarif': row.get('tarif_cie_optimal', ''),
                'nouvelle_option': row.get('option_cie_optimale', ''),
                'economie_mensuelle': economies_cie,
                'economie_annuelle': economies_cie * 12,
                'facilite_mise_en_oeuvre': 'Facile - Demande à CIE'
            })
            recommendations['economies_annuelles_potentielles'] += economies_cie * 12
            recommendations['priorite'] = 'haute' if economies_cie > 15000 else 'moyenne'
        
        # Recommandation HP/HC
        economie_hp_hc = row.get('economie_hp_hc_estimee', 0)
        if economie_hp_hc > 3000:
            recommendations['actions_recommandees'].append({
                'action': 'Passage tarif Heures Pleines/Creuses',
                'condition': 'Si usage majoritairement nocturne possible',
                'economie_mensuelle': economie_hp_hc,
                'economie_annuelle': economie_hp_hc * 12,
                'facilite_mise_en_oeuvre': 'Moyenne - Changement compteur'
            })
            recommendations['economies_annuelles_potentielles'] += economie_hp_hc * 12
        
        # Recommandation optimisation puissance
        if row.get('optimisation_puissance_possible', 0) == 1:
            economie_estimee = 2000  # Estimation conservative
            recommendations['actions_recommandees'].append({
                'action': 'Réduction puissance souscrite',
                'detail': 'Puissance actuelle surdimensionnée',
                'economie_mensuelle': economie_estimee,
                'economie_annuelle': economie_estimee * 12,
                'facilite_mise_en_oeuvre': 'Facile - Demande à CIE'
            })
            recommendations['economies_annuelles_potentielles'] += economie_estimee * 12
        
        return recommendations


def integrate_cie_into_ml_pipeline():
    """Fonction d'intégration complète CIE → ML"""
    
    print("🔗 Intégration CIE dans pipeline ML...")
    
    # Exemple d'utilisation
    integrator = CIEMLIntegrator()
    
    # Charger données existantes
    try:
        df = pd.read_csv('data/processed/combined_dataset.csv')
        print(f"📊 Dataset chargé: {len(df)} lignes")
    except:
        print("⚠️ Dataset non trouvé, création données exemple...")
        # Créer données exemple pour test
        df = pd.DataFrame({
            'mois': ['2023-01', '2023-02', '2023-03'] * 3,
            'kwh_consommes': [850, 920, 1150, 2100, 2250, 2800, 4500, 4200, 5100],
            'montant_fcfa': [127500, 138000, 172500, 315000, 337500, 420000, 675000, 630000, 765000],
            'secteur': ['residential'] * 3 + ['office'] * 3 + ['hotel'] * 3,
            'surface_m2': [150, 150, 150, 200, 200, 200, 500, 500, 500]
        })
    
    # Enrichissement avec tarification CIE
    print("⚡ Enrichissement avec tarification CIE...")
    df_enriched = integrator.enrich_dataset_with_cie_tarifs(df)
    
    # Création features spécialisés CIE
    print("🎯 Création features CIE...")
    df_features = integrator.create_cie_optimization_features(df_enriched)
    
    # Sauvegarde
    output_file = 'data/processed/dataset_with_cie_enrichment.csv'
    df_features.to_csv(output_file, index=False)
    print(f"💾 Dataset enrichi sauvegardé: {output_file}")
    
    # Statistiques d'enrichissement
    print("\n📈 STATISTIQUES ENRICHISSEMENT CIE:")
    print(f"Lignes avec économies CIE détectées: {(df_features['economies_potentielles_cie'] > 0).sum()}")
    print(f"Économies moyennes détectées: {df_features['economies_potentielles_cie'].mean():,.0f} FCFA/mois")
    print(f"Clients avec changement tarif profitable: {(df_features['changement_tranche_profitable'] == 1).sum()}")
    print(f"Clients avec optimisation HP/HC possible: {(df_features['economie_hp_hc_estimee'] > 0).sum()}")
    
    return df_features


# Module de recommandations CIE pour clients
class CIEClientAdvisor:
    """Conseiller CIE pour clients finaux"""
    
    def __init__(self):
        self.integrator = CIEMLIntegrator()
    
    def analyze_client_bill(self, kwh_monthly: float, current_bill_fcfa: float,
                           sector: str, surface_m2: float = None) -> Dict:
        """Analyse complète facture client avec recommandations CIE"""
        
        # Simulation ligne de données client
        client_data = pd.Series({
            'kwh_consommes': kwh_monthly,
            'montant_fcfa': current_bill_fcfa,
            'secteur': sector,
            'surface_m2': surface_m2 or 150
        })
        
        # Enrichissement CIE
        df_temp = pd.DataFrame([client_data])
        df_enriched = self.integrator.enrich_dataset_with_cie_tarifs(df_temp)
        df_features = self.integrator.create_cie_optimization_features(df_enriched)
        
        client_enriched = df_features.iloc[0]
        
        # Générer recommandations
        recommendations = self.integrator.generate_cie_recommendations(client_enriched)
        
        # Analyse détaillée
        analysis = {
            'situation_actuelle': {
                'consommation_kwh': kwh_monthly,
                'facture_actuelle_fcfa': current_bill_fcfa,
                'prix_kwh_moyen': current_bill_fcfa / kwh_monthly,
                'tarif_detecte': client_enriched.get('tarif_cie_optimal', 'Non déterminé')
            },
            'diagnostic_cie': {
                'efficacite_tarifaire': client_enriched.get('efficacite_tarifaire', 1.0),
                'sur_facturation_estimee': max(0, current_bill_fcfa - client_enriched.get('cout_reel_calcule', current_bill_fcfa)),
                'potentiel_economie_pct': client_enriched.get('potentiel_economie_pct', 0)
            },
            'recommandations': recommendations,
            'next_steps': self._generate_next_steps(recommendations)
        }
        
        return analysis
    
    def _generate_next_steps(self, recommendations: Dict) -> List[str]:
        """Générer étapes concrètes pour client"""
        
        steps = []
        
        if recommendations['economies_annuelles_potentielles'] > 50000:  # >50k FCFA/an
            steps.append("1. 📞 Contactez CIE pour audit tarification (gratuit)")
            steps.append("2. 📋 Demandez simulation changement de tranche")
            
        if any('HP/HC' in action.get('action', '') for action in recommendations['actions_recommandees']):
            steps.append("3. ⏰ Évaluez vos possibilités de décalage consommation nocturne")
            steps.append("4. 💡 Demandez devis changement compteur HP/HC")
        
        if any('puissance' in action.get('action', '') for action in recommendations['actions_recommandees']):
            steps.append("5. ⚡ Demandez étude réduction puissance souscrite")
        
        if not steps:
            steps.append("✅ Votre tarification CIE semble déjà optimale")
            steps.append("💡 Focus sur réduction consommation via nos solutions ML")
        
        return steps


# Tests et exemples d'usage
if __name__ == "__main__":
    print("=== TEST INTÉGRATION CIE + ML ===\n")
    
    # Test 1: Intégration complète
    df_result = integrate_cie_into_ml_pipeline()
    
    # Test 2: Conseiller client
    print("\n=== TEST CONSEILLER CLIENT ===")
    advisor = CIEClientAdvisor()
    
    # Exemple client hôtel
    analysis = advisor.analyze_client_bill(
        kwh_monthly=4200,
        current_bill_fcfa=650000,
        sector='hotel',
        surface_m2=500
    )
    
    print(f"Client hôtel - Consommation: 4200 kWh, Facture: 650k FCFA")
    print(f"Économies annuelles potentielles: {analysis['recommandations']['economies_annuelles_potentielles']:,.0f} FCFA")
    print(f"Priorité optimisation: {analysis['recommandations']['priorite']}")
    
    print("\nRecommandations:")
    for i, action in enumerate(analysis['recommandations']['actions_recommandees'], 1):
        print(f"{i}. {action['action']}: {action['economie_mensuelle']:,.0f} FCFA/mois")
    
    print(f"\nÉtapes suivantes:")
    for step in analysis['next_steps']:
        print(f"  {step}")