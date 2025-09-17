"""
💰 Module de Tarification CIE (Compagnie Ivoirienne d'Électricité)
Système complet de calcul des factures et optimisation tarifaire

Données mises à jour : Janvier 2024 (après hausse 10%)
Sources : CIE.ci, ANARE-CI, Gouvernement CI
"""

import json
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import pandas as pd

class CIETarificationSystem:
    """Système de tarification complet CIE"""
    
    def __init__(self):
        self.tarifs = self._load_tarif_structures()
        self.taxes = self._load_tax_structure()
        self.last_update = "2024-01-01"  # Dernière mise à jour tarifs
        
    def _load_tarif_structures(self) -> Dict:
        """Structure tarifaire complète CIE 2024"""
        
        return {
            # TARIFS DOMESTIQUES (Particuliers)
            "domestique_social": {
                "description": "Tarif social pour ménages modestes",
                "puissance_max_kva": 1.1,  # 5A monophasé
                "conditions": {
                    "consommation_max_kwh": 200,
                    "usage": "domestique uniquement",
                    "sur_demande": True
                },
                "tranches": [
                    {"min_kwh": 0, "max_kwh": 50, "prix_fcfa": 69},
                    {"min_kwh": 51, "max_kwh": 110, "prix_fcfa": 79},
                    {"min_kwh": 111, "max_kwh": 200, "prix_fcfa": 87}
                ],
                "abonnement_mensuel": 1247  # FCFA
            },
            
            "domestique_standard": {
                "description": "Tarif domestique standard",
                "puissance_min_kva": 1.1,
                "puissance_max_kva": 36,
                "conditions": {
                    "usage": "domestique",
                    "voltage": "basse_tension"
                },
                "tranches": [
                    {"min_kwh": 0, "max_kwh": 110, "prix_fcfa": 79},
                    {"min_kwh": 111, "max_kwh": 400, "prix_fcfa": 87},
                    {"min_kwh": 401, "max_kwh": float('inf'), "prix_fcfa": 95}
                ],
                "abonnements": {
                    1.1: 1247,   # 5A
                    2.2: 1646,   # 10A  
                    3.3: 2045,   # 15A
                    5.5: 3042,   # 25A
                    11.0: 5537,  # 50A
                    22.0: 10526, # 100A
                    36.0: 17018  # 160A
                }
            },
            
            # TARIFS PROFESSIONNELS
            "professionnel_bt": {
                "description": "Professionnel Basse Tension",
                "puissance_min_kva": 1.1,
                "puissance_max_kva": 250,
                "conditions": {
                    "usage": "professionnel",
                    "voltage": "basse_tension"
                },
                "options": {
                    "simple": {
                        "prix_kwh_fcfa": 101,
                        "abonnements": {
                            3.3: 2870,   # 15A
                            5.5: 4589,   # 25A
                            11.0: 8598,  # 50A
                            22.0: 16627, # 100A
                            36.0: 27044, # 160A
                            50.0: 37462, # 220A
                            100.0: 74923,
                            150.0: 112385,
                            200.0: 149846,
                            250.0: 187308
                        }
                    },
                    "heures_pleines_creuses": {
                        "heures_pleines": {
                            "horaires": ["06:00-22:00"],
                            "prix_kwh_fcfa": 110
                        },
                        "heures_creuses": {
                            "horaires": ["22:00-06:00"],
                            "prix_kwh_fcfa": 79
                        },
                        "abonnements": {  # Majoration de 15% sur tarif simple
                            3.3: 3301,
                            5.5: 5277,
                            11.0: 9888,
                            22.0: 19121,
                            36.0: 31101,
                            50.0: 43081,
                            100.0: 86162,
                            150.0: 129243,
                            200.0: 172323,
                            250.0: 215404
                        }
                    }
                }
            },
            
            "professionnel_mt": {
                "description": "Professionnel Moyenne Tension",
                "puissance_min_kva": 250,
                "puissance_max_kva": 10000,
                "conditions": {
                    "usage": "professionnel",
                    "voltage": "moyenne_tension"
                },
                "prix_kwh_fcfa": 95,
                "abonnement_kva_mois": 2850,  # FCFA par kVA souscrit
                "prime_fixe_mensuelle": 85000
            },
            
            # TARIFS INDUSTRIELS
            "industriel_mt": {
                "description": "Industriel Moyenne Tension",
                "puissance_min_kva": 250,
                "conditions": {
                    "usage": "industriel",
                    "voltage": "moyenne_tension"
                },
                "options": {
                    "standard": {
                        "prix_kwh_fcfa": 89,
                        "abonnement_kva_mois": 2565,
                        "prime_fixe_mensuelle": 85000
                    },
                    "heures_pleines_creuses": {
                        "heures_pleines": {
                            "horaires": ["06:00-22:00"],
                            "prix_kwh_fcfa": 95
                        },
                        "heures_creuses": {
                            "horaires": ["22:00-06:00"],  
                            "prix_kwh_fcfa": 76
                        },
                        "abonnement_kva_mois": 2850,
                        "prime_fixe_mensuelle": 95000
                    }
                }
            },
            
            "industriel_ht": {
                "description": "Industriel Haute Tension",
                "puissance_min_kva": 10000,
                "conditions": {
                    "usage": "industriel",
                    "voltage": "haute_tension"
                },
                "prix_kwh_fcfa": 83,
                "abonnement_kva_mois": 2280,
                "prime_fixe_mensuelle": 170000
            },
            
            # TARIFS SPÉCIAUX
            "courte_utilisation": {
                "description": "Tarif pour usage occasionnel",
                "conditions": {
                    "duree_max_jours": 60,
                    "usage": "temporaire"
                },
                "prix_kwh_fcfa": 125,
                "caution": 50000,
                "frais_branchement": 25000
            },
            
            "prepaye": {
                "description": "Système prépayé (compteur à carte)",
                "prix_kwh_fcfa": 95,
                "frais_carte": 1000,
                "rechargement_min": 5000
            }
        }
    
    def _load_tax_structure(self) -> Dict:
        """Structure des taxes et redevances"""
        
        return {
            "tva": 0.18,  # 18% TVA
            "redevance_eclairage_public": {
                "domestique": 150,      # FCFA/mois
                "professionnel_bt": 300,
                "professionnel_mt": 500,
                "industriel": 750
            },
            "contribution_energie_renouvelable": 0.01,  # 1% du montant HT
            "taxe_municipale": {
                "abidjan": 0.05,  # 5% zones urbaines
                "autres_villes": 0.03,
                "rural": 0.01
            }
        }
    
    def calculate_bill(self, kwh_consumed: float, tarif_type: str, 
                      puissance_kva: float, option: str = "simple",
                      location: str = "abidjan") -> Dict:
        """Calculer facture électrique complète"""
        
        if tarif_type not in self.tarifs:
            raise ValueError(f"Tarif {tarif_type} non trouvé")
        
        tarif_structure = self.tarifs[tarif_type]
        result = {
            "tarif_type": tarif_type,
            "option": option,
            "kwh_consumed": kwh_consumed,
            "puissance_kva": puissance_kva,
            "location": location
        }
        
        # Calcul consommation
        if "tranches" in tarif_structure:
            # Tarif par tranches (domestique)
            cout_consommation = self._calculate_tranche_cost(kwh_consumed, tarif_structure["tranches"])
        elif "options" in tarif_structure:
            # Tarif professionnel avec options
            if option in tarif_structure["options"]:
                option_tarif = tarif_structure["options"][option]
                if "prix_kwh_fcfa" in option_tarif:
                    cout_consommation = kwh_consumed * option_tarif["prix_kwh_fcfa"]
                else:
                    # Heures pleines/creuses (approximation 70%/30%)
                    hp_kwh = kwh_consumed * 0.7
                    hc_kwh = kwh_consumed * 0.3
                    cout_consommation = (
                        hp_kwh * option_tarif["heures_pleines"]["prix_kwh_fcfa"] +
                        hc_kwh * option_tarif["heures_creuses"]["prix_kwh_fcfa"]
                    )
            else:
                raise ValueError(f"Option {option} non disponible pour {tarif_type}")
        else:
            # Tarif simple
            cout_consommation = kwh_consumed * tarif_structure["prix_kwh_fcfa"]
        
        # Abonnement
        abonnement = self._get_abonnement(tarif_structure, puissance_kva, option)
        
        # Prime fixe (MT/HT)
        prime_fixe = tarif_structure.get("prime_fixe_mensuelle", 0)
        if "abonnement_kva_mois" in tarif_structure:
            prime_fixe += puissance_kva * tarif_structure["abonnement_kva_mois"]
        
        # Sous-total HT
        sous_total_ht = cout_consommation + abonnement + prime_fixe
        
        # Taxes
        taxes = self._calculate_taxes(sous_total_ht, tarif_type, location)
        
        # Total TTC
        total_ttc = sous_total_ht + taxes["total_taxes"]
        
        result.update({
            "cout_consommation": round(cout_consommation, 0),
            "abonnement": round(abonnement, 0),
            "prime_fixe": round(prime_fixe, 0),
            "sous_total_ht": round(sous_total_ht, 0),
            "taxes": taxes,
            "total_ttc": round(total_ttc, 0),
            "prix_moyen_kwh": round(total_ttc / kwh_consumed, 1) if kwh_consumed > 0 else 0
        })
        
        return result
    
    def _calculate_tranche_cost(self, kwh: float, tranches: List[Dict]) -> float:
        """Calcul coût par tranches progressives"""
        
        cout_total = 0
        kwh_restant = kwh
        
        for tranche in tranches:
            if kwh_restant <= 0:
                break
                
            tranche_min = tranche["min_kwh"]
            tranche_max = tranche["max_kwh"]
            prix_tranche = tranche["prix_fcfa"]
            
            if tranche_max == float('inf'):
                # Dernière tranche
                kwh_tranche = kwh_restant
            else:
                # Tranche normale
                kwh_tranche = min(kwh_restant, tranche_max - tranche_min + 1)
                if kwh < tranche_min:
                    continue
                if kwh > tranche_max:
                    kwh_tranche = tranche_max - max(tranche_min, kwh - kwh_restant) + 1
            
            cout_total += kwh_tranche * prix_tranche
            kwh_restant -= kwh_tranche
        
        return cout_total
    
    def _get_abonnement(self, tarif_structure: Dict, puissance_kva: float, option: str) -> float:
        """Obtenir coût abonnement selon puissance"""
        
        if "abonnement_mensuel" in tarif_structure:
            return tarif_structure["abonnement_mensuel"]
        
        if "abonnements" in tarif_structure:
            abonnements = tarif_structure["abonnements"]
        elif "options" in tarif_structure and option in tarif_structure["options"]:
            abonnements = tarif_structure["options"][option].get("abonnements", {})
        else:
            return 0
        
        # Trouver puissance correspondante (arrondie supérieure)
        puissances_disponibles = sorted(abonnements.keys())
        for puissance_dispo in puissances_disponibles:
            if puissance_kva <= puissance_dispo:
                return abonnements[puissance_dispo]
        
        # Si puissance > max, prendre le max
        return abonnements[max(puissances_disponibles)]
    
    def _calculate_taxes(self, montant_ht: float, tarif_type: str, location: str) -> Dict:
        """Calcul taxes et redevances"""
        
        taxes = {}
        
        # TVA
        taxes["tva"] = montant_ht * self.taxes["tva"]
        
        # Redevance éclairage public
        if "domestique" in tarif_type:
            taxes["eclairage_public"] = self.taxes["redevance_eclairage_public"]["domestique"]
        elif "professionnel_bt" in tarif_type:
            taxes["eclairage_public"] = self.taxes["redevance_eclairage_public"]["professionnel_bt"]
        elif "professionnel_mt" in tarif_type:
            taxes["eclairage_public"] = self.taxes["redevance_eclairage_public"]["professionnel_mt"]
        elif "industriel" in tarif_type:
            taxes["eclairage_public"] = self.taxes["redevance_eclairage_public"]["industriel"]
        else:
            taxes["eclairage_public"] = 0
        
        # Contribution énergie renouvelable
        taxes["energie_renouvelable"] = montant_ht * self.taxes["contribution_energie_renouvelable"]
        
        # Taxe municipale
        if location in self.taxes["taxe_municipale"]:
            taux_municipal = self.taxes["taxe_municipale"][location]
        else:
            taux_municipal = self.taxes["taxe_municipale"]["autres_villes"]
        taxes["taxe_municipale"] = montant_ht * taux_municipal
        
        taxes["total_taxes"] = sum(taxes.values())
        
        return taxes
    
    def find_optimal_tariff(self, kwh_consumed: float, puissance_kva: float, 
                           usage_type: str = "domestique") -> Dict:
        """Trouver tarif optimal pour une consommation donnée"""
        
        tarifs_eligibles = []
        
        for tarif_name, tarif_info in self.tarifs.items():
            # Vérifier éligibilité
            if usage_type == "domestique" and "domestique" not in tarif_name:
                continue
            if usage_type == "professionnel" and "professionnel" not in tarif_name:
                continue
            if usage_type == "industriel" and "industriel" not in tarif_name:
                continue
                
            # Vérifier puissance
            if "puissance_min_kva" in tarif_info and puissance_kva < tarif_info["puissance_min_kva"]:
                continue
            if "puissance_max_kva" in tarif_info and puissance_kva > tarif_info["puissance_max_kva"]:
                continue
            
            # Calculer coût pour ce tarif
            try:
                if "options" in tarif_info:
                    # Tester toutes les options
                    for option in tarif_info["options"].keys():
                        facture = self.calculate_bill(kwh_consumed, tarif_name, puissance_kva, option)
                        tarifs_eligibles.append({
                            "tarif": tarif_name,
                            "option": option,
                            "cout_total": facture["total_ttc"],
                            "details": facture
                        })
                else:
                    facture = self.calculate_bill(kwh_consumed, tarif_name, puissance_kva)
                    tarifs_eligibles.append({
                        "tarif": tarif_name,
                        "option": "simple",
                        "cout_total": facture["total_ttc"],
                        "details": facture
                    })
            except:
                continue
        
        if not tarifs_eligibles:
            return {"error": "Aucun tarif éligible trouvé"}
        
        # Trier par coût
        tarifs_eligibles.sort(key=lambda x: x["cout_total"])
        
        optimal = tarifs_eligibles[0]
        economies_possibles = []
        
        for i, tarif in enumerate(tarifs_eligibles[1:], 1):
            economie = tarif["cout_total"] - optimal["cout_total"]
            economies_possibles.append({
                "rang": i + 1,
                "tarif": tarif["tarif"],
                "option": tarif["option"],
                "cout": tarif["cout_total"],
                "economie_vs_optimal": economie
            })
        
        return {
            "optimal": optimal,
            "economies_vs_autres": economies_possibles,
            "nb_tarifs_compares": len(tarifs_eligibles)
        }
    
    def simulate_annual_cost(self, monthly_kwh_profile: List[float], 
                            tarif_type: str, puissance_kva: float,
                            option: str = "simple") -> Dict:
        """Simuler coût annuel avec profil mensuel"""
        
        monthly_bills = []
        total_annual = 0
        total_kwh = 0
        
        for month, kwh in enumerate(monthly_kwh_profile, 1):
            bill = self.calculate_bill(kwh, tarif_type, puissance_kva, option)
            bill["month"] = month
            monthly_bills.append(bill)
            total_annual += bill["total_ttc"]
            total_kwh += kwh
        
        return {
            "monthly_bills": monthly_bills,
            "annual_summary": {
                "total_cost_fcfa": round(total_annual, 0),
                "total_kwh": round(total_kwh, 0),
                "average_monthly_cost": round(total_annual / 12, 0),
                "average_monthly_kwh": round(total_kwh / 12, 1),
                "average_price_per_kwh": round(total_annual / total_kwh, 1) if total_kwh > 0 else 0
            },
            "tarif_info": {
                "type": tarif_type,
                "option": option,
                "puissance_kva": puissance_kva
            }
        }
    
    def export_tarifs_to_csv(self, filename: str = "cie_tarifs_2024.csv"):
        """Exporter structure tarifaire vers CSV"""
        
        rows = []
        
        for tarif_name, tarif_info in self.tarifs.items():
            base_row = {
                "tarif_type": tarif_name,
                "description": tarif_info.get("description", ""),
                "puissance_min_kva": tarif_info.get("puissance_min_kva", ""),
                "puissance_max_kva": tarif_info.get("puissance_max_kva", "")
            }
            
            if "tranches" in tarif_info:
                # Tarif par tranches
                for tranche in tarif_info["tranches"]:
                    row = base_row.copy()
                    row.update({
                        "tranche_min_kwh": tranche["min_kwh"],
                        "tranche_max_kwh": tranche["max_kwh"] if tranche["max_kwh"] != float('inf') else "illimité",
                        "prix_kwh_fcfa": tranche["prix_fcfa"],
                        "abonnement_fcfa": tarif_info.get("abonnement_mensuel", "")
                    })
                    rows.append(row)
            elif "prix_kwh_fcfa" in tarif_info:
                # Tarif simple
                row = base_row.copy()
                row.update({
                    "prix_kwh_fcfa": tarif_info["prix_kwh_fcfa"],
                    "abonnement_kva_mois": tarif_info.get("abonnement_kva_mois", ""),
                    "prime_fixe": tarif_info.get("prime_fixe_mensuelle", "")
                })
                rows.append(row)
            elif "options" in tarif_info:
                # Tarif avec options
                for option_name, option_info in tarif_info["options"].items():
                    row = base_row.copy()
                    row.update({
                        "option": option_name,
                        "prix_kwh_fcfa": option_info.get("prix_kwh_fcfa", ""),
                    })
                    if "heures_pleines" in option_info:
                        row["prix_hp_fcfa"] = option_info["heures_pleines"]["prix_kwh_fcfa"]
                        row["prix_hc_fcfa"] = option_info["heures_creuses"]["prix_kwh_fcfa"]
                    rows.append(row)
        
        df = pd.DataFrame(rows)
        df.to_csv(filename, index=False)
        return filename
    
    def export_tarifs_to_json(self, filename: str = "cie_tarifs_2024.json"):
        """Exporter structure tarifaire vers JSON"""
        
        export_data = {
            "last_update": self.last_update,
            "currency": "FCFA",
            "country": "Côte d'Ivoire",
            "operator": "CIE",
            "tarifs": self.tarifs,
            "taxes": self.taxes
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        
        return filename


# Fonctions utilitaires
def quick_tariff_comparison(kwh_monthly: float, puissance_kva: float = 5.5):
    """Comparaison rapide des tarifs pour une consommation donnée"""
    
    cie = CIETarificationSystem()
    
    # Tarifs à comparer
    tarifs_test = [
        ("domestique_standard", "simple"),
        ("professionnel_bt", "simple"),
        ("professionnel_bt", "heures_pleines_creuses")
    ]
    
    results = []
    
    for tarif_type, option in tarifs_test:
        try:
            bill = cie.calculate_bill(kwh_monthly, tarif_type, puissance_kva, option)
            results.append({
                "tarif": tarif_type,
                "option": option,
                "cout_mensuel": bill["total_ttc"],
                "prix_kwh_moyen": bill["prix_moyen_kwh"]
            })
        except:
            continue
    
    return sorted(results, key=lambda x: x["cout_mensuel"])


def calculate_switching_savings(current_bill_fcfa: float, kwh_consumed: float, 
                               current_tariff: str, puissance_kva: float = 5.5):
    """Calculer économies potentielles changement de tarif"""
    
    cie = CIETarificationSystem()
    
    # Trouver tarif optimal
    optimal = cie.find_optimal_tariff(kwh_consumed, puissance_kva)
    
    if "error" in optimal:
        return optimal
    
    optimal_cost = optimal["optimal"]["cout_total"]
    annual_savings = (current_bill_fcfa - optimal_cost) * 12
    
    return {
        "current_monthly_cost": current_bill_fcfa,
        "optimal_monthly_cost": optimal_cost,
        "monthly_savings": current_bill_fcfa - optimal_cost,
        "annual_savings": annual_savings,
        "optimal_tariff": optimal["optimal"]["tarif"],
        "optimal_option": optimal["optimal"]["option"],
        "savings_percentage": (current_bill_fcfa - optimal_cost) / current_bill_fcfa * 100 if current_bill_fcfa > 0 else 0
    }


# Exemple d'utilisation et tests
if __name__ == "__main__":
    # Initialiser le système
    cie = CIETarificationSystem()
    
    # Test 1: Facture domestique standard
    print("=== TEST 1: Maison familiale 1200 kWh ===")
    bill1 = cie.calculate_bill(1200, "domestique_standard", 5.5)
    print(f"Coût total: {bill1['total_ttc']:,.0f} FCFA")
    print(f"Prix moyen: {bill1['prix_moyen_kwh']:.1f} FCFA/kWh")
    
    # Test 2: Bureau professionnel
    print("\n=== TEST 2: Bureau 2500 kWh ===")
    bill2 = cie.calculate_bill(2500, "professionnel_bt", 22.0, "simple")
    print(f"Coût total: {bill2['total_ttc']:,.0f} FCFA")
    
    # Test 3: Comparaison HP/HC
    print("\n=== TEST 3: Comparaison HP/HC ===")
    bill3a = cie.calculate_bill(2500, "professionnel_bt", 22.0, "simple")
    bill3b = cie.calculate_bill(2500, "professionnel_bt", 22.0, "heures_pleines_creuses")
    
    print(f"Tarif simple: {bill3a['total_ttc']:,.0f} FCFA")
    print(f"Tarif HP/HC: {bill3b['total_ttc']:,.0f} FCFA")
    print(f"Économie HP/HC: {bill3a['total_ttc'] - bill3b['total_ttc']:,.0f} FCFA")
    
    # Test 4: Trouver tarif optimal
    print("\n=== TEST 4: Tarif optimal pour 1800 kWh ===")
    optimal = cie.find_optimal_tariff(1800, 11.0, "professionnel")
    if "optimal" in optimal:
        print(f"Tarif optimal: {optimal['optimal']['tarif']} - {optimal['optimal']['option']}")
        print(f"Coût: {optimal['optimal']['cout_total']:,.0f} FCFA")
    
    # Export des tarifs
    print("\n=== EXPORT TARIFS ===")
    csv_file = cie.export_tarifs_to_csv()
    json_file = cie.export_tarifs_to_json()
    print(f"Tarifs exportés: {csv_file}, {json_file}")