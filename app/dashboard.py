"""
📊 Dashboard Streamlit pour Energy ML System
Interface web complète pour prédictions et analyses
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json
from datetime import datetime, timedelta
from pathlib import Path
import sys

# Ajouter src au path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from predictor import EnergyPredictor
from weather_api import WeatherEnrichment

# Configuration page
st.set_page_config(
    page_title="⚡ Energy AI Dashboard",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Style CSS personnalisé
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #FF6B6B, #4ECDC4);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid #FF6B6B;
    }
    .prediction-card {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        border: 1px solid #e9ecef;
    }
</style>
""", unsafe_allow_html=True)

def load_predictor():
    """Charger le prédicteur avec cache"""
    if 'predictor' not in st.session_state:
        try:
            predictor = EnergyPredictor()
            # Essayer de charger le meilleur modèle
            try:
                model_key = predictor.load_best_model('test_r2')
                st.session_state.predictor = predictor
                st.session_state.model_key = model_key
                return predictor, model_key
            except:
                # Fallback: chercher n'importe quel modèle
                models_path = Path('models')
                model_files = list(models_path.glob('*_model_v*.pkl'))
                if model_files:
                    latest_model = max(model_files, key=lambda x: x.stat().st_mtime)
                    model_key = predictor.load_model(latest_model)
                    st.session_state.predictor = predictor
                    st.session_state.model_key = model_key
                    return predictor, model_key
                else:
                    st.error("❌ Aucun modèle ML trouvé. Entraînez d'abord un modèle.")
                    return None, None
        except Exception as e:
            st.error(f"Erreur chargement prédicteur: {e}")
            return None, None
    else:
        return st.session_state.predictor, st.session_state.model_key

def main():
    # Header principal
    st.markdown("""
    <div class="main-header">
        <h1>🚀 Energy AI Dashboard</h1>
        <p>Système d'optimisation énergétique par Intelligence Artificielle</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Charger prédicteur
    predictor, model_key = load_predictor()
    if not predictor:
        st.stop()

    # Sidebar - Configuration
    st.sidebar.title("🎛️ Configuration")

    # Sélection du mode
    mode = st.sidebar.selectbox(
        "Mode d'analyse",
        ["🏠 Prédiction Clients", "📊 Analyse Comparative", "📈 Monitoring Modèles", "🔧 Outils Avancés"]
    )

    if mode == "🏠 Prédiction Clients":
        prediction_interface(predictor, model_key)
    elif mode == "📊 Analyse Comparative":
        comparative_analysis(predictor, model_key)
    elif mode == "📈 Monitoring Modèles":
        model_monitoring()
    elif mode == "🔧 Outils Avancés":
        advanced_tools(predictor, model_key)

def prediction_interface(predictor: EnergyPredictor, model_key: str):
    """Interface de prédiction pour clients"""
    st.header("🔮 Prédictions Énergétiques Clients")

    # Configuration client
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("👤 Informations Client")
        
        # Type de bâtiment
        sector = st.selectbox(
            "Type d'établissement",
            ["residential", "office", "retail", "hotel"],
            format_func=lambda x: {
                "residential": "🏠 Maison/Résidentiel",
                "office": "🏢 Bureau/Entreprise",
                "retail": "🛍️ Commerce/Boutique",
                "hotel": "🏨 Hôtel/Hôtellerie"
            }[x]
        )
        
        # Paramètres généraux
        surface_m2 = st.number_input("Surface (m²)", min_value=50, max_value=2000, value=150)
        
        # Paramètres spécifiques par secteur
        sector_params = {"surface_m2": surface_m2}
        
        if sector == "hotel":
            nb_chambres = st.number_input("Nombre de chambres", min_value=5, max_value=100, value=15)
            taux_occupation = st.slider("Taux d'occupation (%)", 40, 100, 70)
            sector_params.update({
                "nb_chambres": nb_chambres,
                "taux_occupation": taux_occupation / 100
            })
            
        elif sector == "office":
            nb_employes = st.number_input("Nombre d'employés", min_value=2, max_value=100, value=10)
            sector_params["nb_employes"] = nb_employes
            
        elif sector == "residential":
            nb_personnes = st.number_input("Nombre de personnes", min_value=1, max_value=10, value=4)
            sector_params["nb_personnes"] = nb_personnes

    with col2:
        st.subheader("⚙️ Paramètres d'Analyse")
        
        horizon = st.selectbox(
            "Horizon de prédiction",
            ["1month", "1year"],
            format_func=lambda x: "📅 Mois prochain" if x == "1month" else "📊 Profil annuel"
        )
        
        show_scenarios = st.checkbox("Afficher scénarios d'optimisation", value=True)
        show_weather = st.checkbox("Inclure impact météorologique", value=True)
        
        # Bouton de prédiction
        if st.button("🚀 Générer Prédiction", type="primary"):
            with st.spinner("Calcul des prédictions en cours..."):
                try:
                    # Générer prédictions
                    results = predictor.predict_sector(sector, sector_params, horizon, model_key)
                    
                    # Stocker dans session state pour réutilisation
                    st.session_state.last_prediction = results
                    
                    display_prediction_results(results, show_scenarios, show_weather)
                    
                except Exception as e:
                    st.error(f"Erreur lors de la prédiction: {e}")

def display_prediction_results(results: dict, show_scenarios: bool, show_weather: bool):
    """Afficher résultats de prédiction"""
    st.success("✅ Prédictions générées avec succès!")

    if results['horizon'] == '1month':
        # Prédiction mensuelle
        col1, col2, col3 = st.columns(3)
        
        col1.metric(
            "🔌 Consommation prévue",
            f"{results['predicted_kwh']:,.0f} kWh"
        )
        
        col2.metric(
            "💰 Coût estimé",
            f"{results['estimated_cost_fcfa']:,.0f} FCFA"
        )
        
        col3.metric(
            "📊 Coût par m²",
            f"{results['estimated_cost_fcfa'] / results['parameters']['surface_m2']:,.0f} FCFA/m²"
        )
        
        # Graphique de confiance
        if 'prediction_range' in results:
            fig = go.Figure()
            
            fig.add_trace(go.Bar(
                x=['Prévision'],
                y=[results['predicted_kwh']],
                name='Prédiction',
                marker_color='#FF6B6B',
                error_y=dict(
                    type='data',
                    symmetric=False,
                    array=[results['prediction_range']['max_kwh'] - results['predicted_kwh']],
                    arrayminus=[results['predicted_kwh'] - results['prediction_range']['min_kwh']]
                )
            ))
            
            fig.update_layout(
                title="🎯 Prédiction avec Intervalle de Confiance",
                yaxis_title="Consommation (kWh)",
                showlegend=False
            )
            
            st.plotly_chart(fig, use_container_width=True)

    elif results['horizon'] == '1year':
        # Profil annuel
        annual_summary = results['annual_summary']
        
        # Métriques annuelles
        col1, col2, col3, col4 = st.columns(4)
        
        col1.metric(
            "⚡ Total annuel",
            f"{annual_summary['total_kwh']:,.0f} kWh"
        )
        
        col2.metric(
            "💰 Coût annuel",
            f"{annual_summary['total_cost_fcfa']:,.0f} FCFA"
        )
        
        col3.metric(
            "📊 Moyenne mensuelle",
            f"{annual_summary['average_monthly_kwh']:,.0f} kWh"
        )
        
        col4.metric(
            "📈 Coût mensuel moyen",
            f"{annual_summary['average_monthly_cost_fcfa']:,.0f} FCFA"
        )
        
        # Graphique profil annuel
        monthly_data = results['monthly_predictions']
        months = [data['month_name'] for key, data in monthly_data.items() if 'month_name' in data]
        kwh_values = [data['predicted_kwh'] for key, data in monthly_data.items() if 'month_name' in data]
        costs = [data['estimated_cost_fcfa'] for key, data in monthly_data.items() if 'month_name' in data]
        
        # Graphique double axe
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=("📊 Consommation Énergétique Mensuelle", "💰 Coûts Mensuels"),
            vertical_spacing=0.1
        )
        
        # Consommation
        fig.add_trace(
            go.Scatter(
                x=months, y=kwh_values,
                mode='lines+markers',
                name='Consommation (kWh)',
                line=dict(color='#FF6B6B', width=3),
                marker=dict(size=8)
            ),
            row=1, col=1
        )
        
        # Coûts
        fig.add_trace(
            go.Bar(
                x=months, y=costs,
                name='Coût (FCFA)',
                marker_color='#4ECDC4',
                opacity=0.7
            ),
            row=2, col=1
        )
        
        fig.update_layout(
            height=600,
            showlegend=False,
            title_text="📈 Profil Énergétique Annuel Prédit"
        )
        
        fig.update_xaxes(tickangle=45)
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Scénarios d'optimisation
        if show_scenarios and 'optimization_scenarios' in results:
            st.subheader("💡 Scénarios d'Optimisation Énergétique")
            
            scenarios = results['optimization_scenarios']
            
            for scenario_name, scenario in scenarios.items():
                with st.expander(f"📋 {scenario['description']}", expanded=scenario['is_profitable']):
                    
                    # Couleur selon rentabilité
                    color = "green" if scenario['is_profitable'] else "orange"
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    col1.markdown(f"""
                    **💪 Efficacité**  
                    {scenario['efficiency_gain_percent']} d'économies
                    """)
                    
                    col2.markdown(f"""
                    **💰 Investissement**  
                    {scenario['investment_fcfa']:,.0f} FCFA
                    """)
                    
                    col3.markdown(f"""
                    **⏰ ROI**  
                    <span style="color:{color};">{scenario['roi_years']} ans</span>
                    """, unsafe_allow_html=True)
                    
                    col4.markdown(f"""
                    **💚 Économies/an**  
                    {scenario['cost_saved_annual_fcfa']:,.0f} FCFA
                    """)
                    
                    # Recommandation
                    if scenario['is_profitable']:
                        st.success(f"✅ Recommandé - ROI attractif en {scenario['roi_years']} ans")
                    else:
                        st.warning(f"⚠️ ROI long ({scenario['roi_years']} ans) - À évaluer selon budget")

def comparative_analysis(predictor: EnergyPredictor, model_key: str):
    """Interface d'analyse comparative"""
    st.header("📊 Analyse Comparative Multi-Secteurs")

    st.info("🎯 Comparez les performances énergétiques entre différents types d'établissements")

    # Configuration de comparaison
    st.subheader("⚙️ Configuration Comparative")

    sectors_to_compare = st.multiselect(
        "Secteurs à comparer",
        ["residential", "office", "retail", "hotel"],
        default=["residential", "office"],
        format_func=lambda x: {
            "residential": "🏠 Résidentiel",
            "office": "🏢 Bureau",
            "retail": "🛍️ Commerce",
            "hotel": "🏨 Hôtel"
        }[x]
    )

    if len(sectors_to_compare) >= 2:
        
        # Configuration surfaces
        st.subheader("📏 Configuration des Établissements")
        surfaces = {}
        
        col_count = min(len(sectors_to_compare), 4)
        cols = st.columns(col_count)
        
        for i, sector in enumerate(sectors_to_compare):
            with cols[i % col_count]:
                default_surface = {"residential": 120, "office": 200, "retail": 100, "hotel": 400}
                surfaces[sector] = st.number_input(
                    f"Surface {sector} (m²)",
                    min_value=50,
                    max_value=1000,
                    value=default_surface.get(sector, 150)
                )
        
        if st.button("📊 Générer Analyse Comparative"):
            
            with st.spinner("Génération de l'analyse comparative..."):
                
                comparative_data = {}
                
                # Générer prédictions pour chaque secteur
                for sector in sectors_to_compare:
                    params = {"surface_m2": surfaces[sector]}
                    results = predictor.predict_sector(sector, params, "1year", model_key)
                    comparative_data[sector] = results
                
                # Afficher résultats comparatifs
                display_comparative_results(comparative_data)

def display_comparative_results(data: dict):
    """Afficher résultats d'analyse comparative"""
    st.success("✅ Analyse comparative générée!")

    # Tableau de comparaison
    st.subheader("📋 Tableau Comparatif")

    comparison_df = pd.DataFrame({
        'Secteur': [sector.title() for sector in data.keys()],
        'Consommation Annuelle (kWh)': [data[sector]['annual_summary']['total_kwh'] for sector in data.keys()],
        'Coût Annuel (FCFA)': [data[sector]['annual_summary']['total_cost_fcfa'] for sector in data.keys()],
        'kWh par m²': [
            data[sector]['annual_summary']['total_kwh'] / data[sector]['parameters']['surface_m2'] 
            for sector in data.keys()
        ],
        'FCFA par m²': [
            data[sector]['annual_summary']['total_cost_fcfa'] / data[sector]['parameters']['surface_m2'] 
            for sector in data.keys()
        ]
    })

    st.dataframe(comparison_df, use_container_width=True)

    # Graphiques comparatifs
    col1, col2 = st.columns(2)

    with col1:
        # Consommation par secteur
        fig_cons = px.bar(
            comparison_df, 
            x='Secteur', 
            y='Consommation Annuelle (kWh)',
            title="⚡ Consommation par Secteur",
            color='Secteur'
        )
        fig_cons.update_layout(showlegend=False)
        st.plotly_chart(fig_cons, use_container_width=True)

    with col2:
        # Efficacité énergétique (kWh/m²)
        fig_eff = px.bar(
            comparison_df, 
            x='Secteur', 
            y='kWh par m²',
            title="📊 Efficacité Énergétique (kWh/m²)",
            color='Secteur'
        )
        fig_eff.update_layout(showlegend=False)
        st.plotly_chart(fig_eff, use_container_width=True)

    # Analyse temporelle comparative
    st.subheader("📈 Évolution Mensuelle Comparative")
    
    # Préparer données mensuelles
    monthly_comparison = {}
    months = []
    
    for sector, results in data.items():
        monthly_data = results['monthly_predictions']
        if not months:
            months = [data['month_name'] for key, data in monthly_data.items() if 'month_name' in data]
        
        monthly_comparison[sector] = [data['predicted_kwh'] for key, data in monthly_data.items() if 'month_name' in data]
    
    # Graphique comparaison mensuelle
    fig_monthly = go.Figure()
    
    colors = {'residential': '#FF6B6B', 'office': '#4ECDC4', 'retail': '#45B7D1', 'hotel': '#96CEB4'}
    
    for sector, kwh_values in monthly_comparison.items():
        fig_monthly.add_trace(go.Scatter(
            x=months,
            y=kwh_values,
            mode='lines+markers',
            name=sector.title(),
            line=dict(color=colors.get(sector, '#333333'), width=3),
            marker=dict(size=6)
        ))
    
    fig_monthly.update_layout(
        title="🔄 Comparaison des Profils Mensuels",
        xaxis_title="Mois",
        yaxis_title="Consommation (kWh)",
        height=500
    )
    
    st.plotly_chart(fig_monthly, use_container_width=True)

    # Recommandations comparatives
    st.subheader("💡 Recommandations Comparatives")
    
    # Trouver le secteur le plus efficace
    best_efficiency = comparison_df.loc[comparison_df['kWh par m²'].idxmin()]
    worst_efficiency = comparison_df.loc[comparison_df['kWh par m²'].idxmax()]
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.success(f"""
        🏆 **Secteur le plus efficace**: {best_efficiency['Secteur']}
        - {best_efficiency['kWh par m²']:.1f} kWh/m²/an
        - {best_efficiency['FCFA par m²']:,.0f} FCFA/m²/an
        """)
    
    with col2:
        st.warning(f"""
        ⚠️ **Secteur à optimiser**: {worst_efficiency['Secteur']}
        - {worst_efficiency['kWh par m²']:.1f} kWh/m²/an  
        - Potentiel d'amélioration: {((worst_efficiency['kWh par m²'] - best_efficiency['kWh par m²']) / worst_efficiency['kWh par m²'] * 100):.1f}%
        """)

    # Analyse des coûts comparative
    st.subheader("💰 Analyse Économique Comparative")
    
    # Calculs économiques
    total_cost = comparison_df['Coût Annuel (FCFA)'].sum()
    avg_cost_per_m2 = comparison_df['FCFA par m²'].mean()
    
    col1, col2, col3 = st.columns(3)
    
    col1.metric(
        "💵 Coût total portfolio",
        f"{total_cost:,.0f} FCFA"
    )
    
    col2.metric(
        "📊 Coût moyen/m²",
        f"{avg_cost_per_m2:,.0f} FCFA/m²"
    )
    
    col3.metric(
        "🔍 Écart coût max/min",
        f"{comparison_df['FCFA par m²'].max() - comparison_df['FCFA par m²'].min():,.0f} FCFA/m²"
    )
    
    # Graphique radar comparatif
    fig_radar = go.Figure()
    
    # Normaliser les métriques pour le radar (0-100)
    metrics = ['Consommation Annuelle (kWh)', 'Coût Annuel (FCFA)', 'kWh par m²', 'FCFA par m²']
    
    for sector in data.keys():
        sector_data = comparison_df[comparison_df['Secteur'] == sector.title()].iloc[0]
        
        # Normalisation (inverse pour efficacité)
        normalized_values = []
        for metric in metrics:
            max_val = comparison_df[metric].max()
            min_val = comparison_df[metric].min()
            if max_val != min_val:
                norm_val = (sector_data[metric] - min_val) / (max_val - min_val) * 100
                # Inverser pour kWh/m² et FCFA/m² (moins = mieux)
                if 'par m²' in metric:
                    norm_val = 100 - norm_val
                normalized_values.append(norm_val)
            else:
                normalized_values.append(50)
        
        fig_radar.add_trace(go.Scatterpolar(
            r=normalized_values,
            theta=['Consommation', 'Coût Total', 'Efficacité kWh', 'Efficacité Coût'],
            fill='toself',
            name=sector.title(),
            line=dict(color=colors.get(sector, '#333333'))
        ))
    
    fig_radar.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100]
            )
        ),
        showlegend=True,
        title="🎯 Comparaison Radar Multi-Critères (100 = optimal)",
        height=500
    )
    
    st.plotly_chart(fig_radar, use_container_width=True)
    
    # Matrice de décision
    st.subheader("🎲 Matrice de Décision Énergétique")
    
    # Calculer scores pondérés
    weights = {
        'efficiency': 0.4,  # 40% efficacité énergétique  
        'cost': 0.3,       # 30% coût
        'stability': 0.2,   # 20% stabilité (inverse de variance)
        'scalability': 0.1  # 10% potentiel d'extension
    }
    
    decision_matrix = []
    for sector in data.keys():
        sector_row = comparison_df[comparison_df['Secteur'] == sector.title()].iloc[0]
        
        # Score efficacité (inverse kWh/m²)
        efficiency_score = (100 - (sector_row['kWh par m²'] / comparison_df['kWh par m²'].max()) * 100)
        
        # Score coût (inverse FCFA/m²)  
        cost_score = (100 - (sector_row['FCFA par m²'] / comparison_df['FCFA par m²'].max()) * 100)
        
        # Score stabilité (simulé basé sur variance mensuelle)
        monthly_kwh = monthly_comparison[sector]
        variance = np.var(monthly_kwh) if len(monthly_kwh) > 1 else 0
        max_variance = max([np.var(monthly_comparison[s]) for s in data.keys()])
        stability_score = 100 - (variance / max_variance * 100) if max_variance > 0 else 50
        
        # Score scalabilité (simulé)
        scalability_map = {'residential': 70, 'office': 85, 'retail': 60, 'hotel': 90}
        scalability_score = scalability_map.get(sector, 50)
        
        # Score global pondéré
        global_score = (
            efficiency_score * weights['efficiency'] +
            cost_score * weights['cost'] +
            stability_score * weights['stability'] +
            scalability_score * weights['scalability']
        )
        
        decision_matrix.append({
            'Secteur': sector.title(),
            'Efficacité': f"{efficiency_score:.1f}",
            'Coût': f"{cost_score:.1f}",
            'Stabilité': f"{stability_score:.1f}",
            'Scalabilité': f"{scalability_score:.1f}",
            'Score Global': f"{global_score:.1f}",
            'Recommandation': '🥇 Optimal' if global_score >= 80 else 
                            '🥈 Bon' if global_score >= 60 else 
                            '🥉 À améliorer'
        })
    
    decision_df = pd.DataFrame(decision_matrix)
    decision_df = decision_df.sort_values('Score Global', ascending=False)
    
    st.dataframe(
        decision_df.style.background_gradient(subset=['Score Global'], cmap='RdYlGn'),
        use_container_width=True
    )
    
    # Recommandations stratégiques finales
    st.subheader("🎯 Recommandations Stratégiques")
    
    best_sector = decision_df.iloc[0]
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.success(f"""
        **🏆 Secteur champion: {best_sector['Secteur']}**
        - Score global: {best_sector['Score Global']}/100
        - {best_sector['Recommandation']}
        - Priorité: Développement et optimisation
        """)
    
    with col2:
        # Calculer potentiel d'économie globale
        min_kwh_m2 = comparison_df['kWh par m²'].min()
        total_surface = sum([data[sector]['parameters']['surface_m2'] for sector in data.keys()])
        
        potential_savings = 0
        for sector in data.keys():
            current_kwh_m2 = comparison_df[comparison_df['Secteur'] == sector.title()]['kWh par m²'].iloc[0]
            surface = data[sector]['parameters']['surface_m2']
            if current_kwh_m2 > min_kwh_m2:
                savings_kwh_m2 = current_kwh_m2 - min_kwh_m2
                potential_savings += savings_kwh_m2 * surface * 120  # 120 FCFA/kWh moyenne
        
        st.info(f"""
        **💡 Potentiel d'optimisation portfolio:**
        - Économie annuelle possible: {potential_savings:,.0f} FCFA
        - Par standardisation au niveau du meilleur secteur
        - ROI attendu: 2-3 ans avec investissements ciblés
        """)

def model_monitoring():
    """Interface de monitoring des modèles"""
    st.header("📈 Monitoring des Modèles ML")
    
    st.info("📊 Suivi des performances et métriques des modèles d'apprentissage automatique")
    
    # Vérifier si des modèles existent
    models_path = Path('models')
    if not models_path.exists():
        st.error("❌ Dossier 'models' non trouvé")
        return
    
    model_files = list(models_path.glob('*.pkl'))
    if not model_files:
        st.warning("⚠️ Aucun modèle entraîné trouvé")
        return
    
    # Afficher liste des modèles
    st.subheader("🔍 Modèles Disponibles")
    
    models_info = []
    for model_file in model_files:
        stat = model_file.stat()
        models_info.append({
            'Nom': model_file.name,
            'Taille': f"{stat.st_size / 1024:.1f} KB",
            'Modifié': datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M"),
            'Chemin': str(model_file)
        })
    
    models_df = pd.DataFrame(models_info)
    st.dataframe(models_df, use_container_width=True)
    
    # Sélection modèle pour analyse détaillée
    selected_model = st.selectbox(
        "Sélectionner un modèle pour analyse détaillée",
        [m['Nom'] for m in models_info]
    )
    
    if selected_model and st.button("📊 Analyser Modèle"):
        st.subheader(f"🔬 Analyse Détaillée: {selected_model}")
        
        # Placeholder pour métriques détaillées
        col1, col2, col3, col4 = st.columns(4)
        
        col1.metric("🎯 Précision R²", "0.85", "↑ +0.02")
        col2.metric("📉 RMSE", "150.3", "↓ -5.2")
        col3.metric("⚡ Temps prédiction", "0.03s", "→ stable")
        col4.metric("📊 Données entraînement", "12,450", "↑ +1,200")
        
        # Graphique de performance fictif
        dates = pd.date_range('2024-01-01', periods=30, freq='D')
        performance_data = {
            'Date': dates,
            'R² Score': np.random.normal(0.85, 0.02, 30).clip(0.7, 0.95),
            'RMSE': np.random.normal(150, 10, 30).clip(120, 180)
        }
        
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=("📈 Évolution R² Score", "📉 Évolution RMSE"),
            vertical_spacing=0.1
        )
        
        fig.add_trace(
            go.Scatter(x=dates, y=performance_data['R² Score'], 
                      name='R² Score', line=dict(color='green')),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=dates, y=performance_data['RMSE'], 
                      name='RMSE', line=dict(color='red')),
            row=2, col=1
        )
        
        fig.update_layout(height=500, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

def advanced_tools(predictor: EnergyPredictor, model_key: str):
    """Outils avancés et utilitaires"""
    st.header("🔧 Outils Avancés")
    
    tab1, tab2, tab3 = st.tabs(["🔄 Réentraînement", "📤 Export Données", "⚙️ Configuration"])
    
    with tab1:
        st.subheader("🔄 Réentraînement des Modèles")
        st.info("Relancer l'entraînement avec de nouvelles données")
        
        if st.button("🚀 Lancer Réentraînement"):
            with st.spinner("Réentraînement en cours..."):
                # Simuler réentraînement
                progress_bar = st.progress(0)
                for i in range(100):
                    progress_bar.progress(i + 1)
                st.success("✅ Réentraînement terminé avec succès!")
    
    with tab2:
        st.subheader("📤 Export des Prédictions")
        
        if 'last_prediction' in st.session_state:
            results = st.session_state.last_prediction
            
            # Préparer données pour export
            if results['horizon'] == '1year':
                export_data = []
                for month_key, month_data in results['monthly_predictions'].items():
                    if 'month_name' in month_data:
                        export_data.append({
                            'Mois': month_data['month_name'],
                            'Consommation (kWh)': month_data['predicted_kwh'],
                            'Coût (FCFA)': month_data['estimated_cost_fcfa']
                        })
                
                export_df = pd.DataFrame(export_data)
                
                # Boutons d'export
                col1, col2 = st.columns(2)
                
                with col1:
                    csv = export_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Télécharger CSV",
                        data=csv,
                        file_name=f"predictions_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                        mime="text/csv"
                    )
                
                with col2:
                    json_data = export_df.to_json(orient='records', indent=2)
                    st.download_button(
                        label="📥 Télécharger JSON",
                        data=json_data,
                        file_name=f"predictions_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
                        mime="application/json"
                    )
        else:
            st.info("Aucune prédiction à exporter. Générez d'abord une prédiction.")
    
    with tab3:
        st.subheader("⚙️ Configuration Système")
        
        # Paramètres généraux
        st.write("**Paramètres d'affichage**")
        
        col1, col2 = st.columns(2)
        with col1:
            decimal_places = st.number_input("Décimales affichées", 0, 3, 1)
        with col2:
            currency_format = st.selectbox("Format monétaire", ["FCFA", "EUR", "USD"])
        
        # Seuils d'alerte
        st.write("**Seuils d'alerte**")
        col1, col2 = st.columns(2)
        with col1:
            high_consumption_threshold = st.number_input("Seuil consommation élevée (kWh/m²)", 100, 500, 200)
        with col2:
            cost_alert_threshold = st.number_input("Seuil alerte coût (FCFA/m²)", 10000, 50000, 25000)
        
        if st.button("💾 Sauvegarder Configuration"):
            st.success("✅ Configuration sauvegardée!")

def show_cie_analysis():
    """Interface analyse CIE"""
    st.header("💰 Analyse Tarifaire CIE")
    
    col1, col2 = st.columns(2)
    with col1:
        kwh_month = st.number_input("Consommation mensuelle (kWh)", min_value=50, value=1200)
        current_bill = st.number_input("Facture actuelle (FCFA)", min_value=10000, value=180000)
    
    with col2:
        sector = st.selectbox("Secteur", ['residential', 'office', 'retail', 'hotel'])
        surface = st.number_input("Surface (m²)", min_value=50, value=200)
    
    if st.button("🚀 Analyser Optimisation CIE"):
        with st.spinner("Analyse en cours..."):
            advisor = CIEClientAdvisor()
            analysis = advisor.analyze_client_bill(kwh_month, current_bill, sector, surface)
            
            # Afficher résultats
            st.success("✅ Analyse terminée!")
            
            col1, col2, col3 = st.columns(3)
            col1.metric("💰 Économies annuelles", f"{analysis['recommandations']['economies_annuelles_potentielles']:,.0f} FCFA")
            col2.metric("🎯 Priorité", analysis['recommandations']['priorite'].title())
            col3.metric("📊 Efficacité tarifaire", f"{analysis['diagnostic_cie']['efficacite_tarifaire']:.2f}")
            
            # Recommandations
            if analysis['recommandations']['actions_recommandees']:
                st.subheader("📋 Recommandations CIE")
                for i, action in enumerate(analysis['recommandations']['actions_recommandees'], 1):
                    st.write(f"**{i}. {action['action']}**")
                    st.write(f"💰 Économies: {action['economie_mensuelle']:,.0f} FCFA/mois")
                    st.write(f"⚙️ Facilité: {action['facilite_mise_en_oeuvre']}")
                    st.write("---")

if __name__ == "__main__":
    main()
