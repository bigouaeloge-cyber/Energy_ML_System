#!/usr/bin/env python3
"""
Script de setup initial du module CIE
"""
import sys
from pathlib import Path

# Ajouter src au path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

def setup_cie_files():
    """Générer tous les fichiers CIE"""
    try:
        from cie_tarification import CIETarificationSystem
        
        print("🔄 Génération fichiers CIE...")
        cie = CIETarificationSystem()
        
        # Créer dossier config s'il n'existe pas
        config_dir = Path('config')
        config_dir.mkdir(exist_ok=True)
        
        # Générer fichiers
        csv_file = cie.export_tarifs_to_csv("config/cie_tarifs_2024.csv")
        json_file = cie.export_tarifs_to_json("config/cie_tarifs_2024.json")
        
        print("✅ Fichiers CIE générés:")
        print(f"  📊 {csv_file}")
        print(f"  📋 {json_file}")
        
        return True
        
    except Exception as e:
        print(f"❌ Erreur génération CIE: {e}")
        return False

if __name__ == "__main__":
    success = setup_cie_files()
    if success:
        print("\n🎉 Module CIE configuré avec succès!")
    else:
        print("\n⚠️ Problème configuration CIE")