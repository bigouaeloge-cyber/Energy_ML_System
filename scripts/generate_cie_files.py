#!/usr/bin/env python3
"""
Script de génération automatique des fichiers tarifs CIE
Usage: python generate_cie_files.py
"""

from cie_tarification import CIETarificationSystem
import pandas as pd

def generate_all_cie_files():
    """Générer tous les fichiers tarifs CIE"""
    
    print("🔄 Génération fichiers tarifs CIE...")
    
    # Initialiser système CIE
    cie = CIETarificationSystem()
    
    # 1. Fichiers de base (CSV + JSON)
    csv_file = cie.export_tarifs_to_csv("config/cie_tarifs_2024.csv")
    json_file = cie.export_tarifs_to_json("config/cie_tarifs_2024.json")
    
    # 2. Fichier abonnements détaillé
    abonnements_data = []
    
    # Domestique
    for kva, fcfa in cie.tarifs["domestique_standard"]["abonnements"].items():
        amperage = int(kva * 4.55) if kva <= 11 else int(kva * 1.44)  # Approximation A = kVA/0.22 ou kVA/0.38
        voltage = "monophasé" if kva <= 11 else "triphasé"
        
        abonnements_data.append({
            "puissance_kva": kva,
            "amperage": amperage,
            "voltage": voltage,
            "domestique_fcfa": fcfa,
            "professionnel_bt_fcfa": cie.tarifs["professionnel_bt"]["options"]["simple"]["abonnements"].get(kva, ""),
            "professionnel_bt_hphc_fcfa": cie.tarifs["professionnel_bt"]["options"]["heures_pleines_creuses"]["abonnements"].get(kva, ""),
            "usage_type": "domestique_standard" if kva <= 36 else "professionnel",
            "installation_type": "standard" if kva <= 5.5 else "renforcé" if kva <= 11 else "triphasé" if kva <= 250 else "moyenne_tension"
        })
    
    # Professionnel (puissances supérieures)
    for kva, fcfa in cie.tarifs["professionnel_bt"]["options"]["simple"]["abonnements"].items():
        if kva not in [row["puissance_kva"] for row in abonnements_data]:
            amperage = int(kva * 1.44)  # Triphasé
            
            abonnements_data.append({
                "puissance_kva": kva,
                "amperage": amperage,
                "voltage": "triphasé",
                "domestique_fcfa": "",
                "professionnel_bt_fcfa": fcfa,
                "professionnel_bt_hphc_fcfa": cie.tarifs["professionnel_bt"]["options"]["heures_pleines_creuses"]["abonnements"].get(kva, ""),
                "usage_type": "professionnel" if kva <= 250 else "professionnel_mt",
                "installation_type": "triphasé" if kva <= 250 else "moyenne_tension"
            })
    
    df_abonnements = pd.DataFrame(abonnements_data)
    df_abonnements.to_csv("config/cie_abonnements_2024.csv", index=False)
    
    # 3. Fichier comparaison tarifs (pour ML)
    comparaison_data = []
    
    kwh_tests = [100, 300, 500, 800, 1200, 1800, 2500, 4000, 6000, 10000]
    
    for kwh in kwh_tests:
        for tarif_name in ["domestique_standard", "professionnel_bt"]:
            try:
                # Tarif simple
                bill_simple = cie.calculate_bill(kwh, tarif_name, 11.0, "simple")
                comparaison_data.append({
                    "kwh": kwh,
                    "tarif": tarif_name,
                    "option": "simple",
                    "cout_total": bill_simple["total_ttc"],
                    "prix_kwh_moyen": bill_simple["prix_moyen_kwh"]
                })
                
                # Tarif HP/HC si disponible
                if tarif_name == "professionnel_bt":
                    bill_hphc = cie.calculate_bill(kwh, tarif_name, 11.0, "heures_pleines_creuses")
                    comparaison_data.append({
                        "kwh": kwh,
                        "tarif": tarif_name,
                        "option": "heures_pleines_creuses",
                        "cout_total": bill_hphc["total_ttc"],
                        "prix_kwh_moyen": bill_hphc["prix_moyen_kwh"]
                    })
            except:
                continue
    
    df_comparaison = pd.DataFrame(comparaison_data)
    df_comparaison.to_csv("config/cie_comparaison_tarifs.csv", index=False)
    
    # 4. Fichier seuils optimisation (pour ML)
    seuils_data = []
    
    # Seuils domestiques
    seuils_data.extend([
        {"type": "domestique", "seuil_kwh": 110, "description": "Passage tranche 1 → 2 (79→87 FCFA)", "impact": "Augmentation 10%"},
        {"type": "domestique", "seuil_kwh": 400, "description": "Passage tranche 2 → 3 (87→95 FCFA)", "impact": "Augmentation 9%"},
        {"type": "domestique", "seuil_kwh": 1000, "description": "Seuil passage professionnel potentiel", "impact": "Évaluation cas par cas"}
    ])
    
    # Seuils professionnels
    seuils_data.extend([
        {"type": "professionnel", "seuil_kwh": 2000, "description": "Rentabilité HP/HC généralement atteinte", "impact": "Économies possibles"},
        {"type": "professionnel", "seuil_kwh": 5000, "description": "Passage MT souvent rentable", "impact": "Réduction tarif possible"}
    ])
    
    df_seuils = pd.DataFrame(seuils_data)
    df_seuils.to_csv("config/cie_seuils_optimisation.csv", index=False)
    
    print("✅ Fichiers générés:")
    print(f"  📊 {csv_file}")
    print(f"  📋 {json_file}")
    print(f"  💰 config/cie_abonnements_2024.csv")
    print(f"  📈 config/cie_comparaison_tarifs.csv")
    print(f"  🎯 config/cie_seuils_optimisation.csv")
    
    return {
        "tarifs_csv": csv_file,
        "tarifs_json": json_file,
        "abonnements_csv": "config/cie_abonnements_2024.csv",
        "comparaison_csv": "config/cie_comparaison_tarifs.csv",
        "seuils_csv": "config/cie_seuils_optimisation.csv"
    }

if __name__ == "__main__":
    files = generate_all_cie_files()
    print(f"\n🎉 {len(files)} fichiers CIE générés avec succès!")