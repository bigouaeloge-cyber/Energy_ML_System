#!/usr/bin/env python3
"""
Test intégration module CIE
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent / 'src'))

def test_cie_integration():
    """Tester intégration CIE complète"""
    
    print("🧪 Test intégration module CIE...")
    
    # Test 1: Import modules
    try:
        from cie_tarification import CIETarificationSystem
        from cie_ml_integration import CIEMLIntegrator, CIEClientAdvisor
        print("✅ Imports CIE OK")
    except ImportError as e:
        print(f"❌ Erreur import: {e}")
        return False
    
    # Test 2: Calcul facture
    try:
        cie = CIETarificationSystem()
        bill = cie.calculate_bill(1200, "domestique_standard", 5.5)
        print(f"✅ Calcul facture OK: {bill['total_ttc']:,.0f} FCFA")
    except Exception as e:
        print(f"❌ Erreur calcul: {e}")
        return False
    
    # Test 3: Conseiller client
    try:
        advisor = CIEClientAdvisor()
        analysis = advisor.analyze_client_bill(1200, 180000, 'office', 200)
        economies = analysis['recommandations']['economies_annuelles_potentielles']
        print(f"✅ Conseiller CIE OK: {economies:,.0f} FCFA économies/an")
    except Exception as e:
        print(f"❌ Erreur conseiller: {e}")
        return False
    
    # Test 4: Intégration ML
    try:
        import pandas as pd
        integrator = CIEMLIntegrator()
        
        # Dataset test
        df_test = pd.DataFrame({
            'kwh_consommes': [800, 1200, 2500],
            'montant_fcfa': [120000, 180000, 375000],
            'secteur': ['residential', 'office', 'hotel'],
            'surface_m2': [120, 200, 400]
        })
        
        df_enriched = integrator.enrich_dataset_with_cie_tarifs(df_test)
        new_columns = len(df_enriched.columns) - len(df_test.columns)
        print(f"✅ Enrichissement ML OK: +{new_columns} colonnes CIE ajoutées")
        
    except Exception as e:
        print(f"❌ Erreur intégration ML: {e}")
        return False
    
    print("\n🎉 INTÉGRATION CIE COMPLÈTE RÉUSSIE!")
    return True

if __name__ == "__main__":
    test_cie_integration()