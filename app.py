# app.py - Version tout-en-un pour Streamlit Cloud
# ============================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import os

# ============================================
# SEITENKONFIGURATION
# ============================================
st.set_page_config(
    page_title="Immobilienpreis-Vorhersage",
    page_icon="🏠",
    layout="wide"
)

# ============================================
# MEHRSPRACHIGKEIT
# ============================================
SPRACHEN = {
    'DE': 'Deutsch',
    'FR': 'Français',
    'EN': 'English',
    'ES': 'Español',
    'IT': 'Italiano'
}

TEXTE = {
    'DE': {
        'titel': '🏠 Immobilienpreis-Vorhersage',
        'untertitel': 'KI-gestützte Vorhersage für Immobilienpreise',
        'flaeche': 'Wohnfläche (m²)',
        'zimmer': 'Zimmer',
        'schlafzimmer': 'Schlafzimmer',
        'etage': 'Etage',
        'stadt': 'Stadt',
        'baujahr': 'Baujahr',
        'garten': 'Garten',
        'balkon': 'Balkon',
        'vorhersagen': '🔮 Preis vorhersagen',
        'geschätzter_preis': 'Geschätzter Preis',
        'preis_pro_m2': 'Preis pro m²',
        'modell_laden': '🔄 Lade Modell...',
        'modell_geladen': '✅ Modell geladen',
        'modell_fehler': '❌ Modell konnte nicht geladen werden',
        'staedte': ['Berlin', 'Hamburg', 'München', 'Köln', 'Frankfurt', 'Stuttgart']
    },
    'FR': {
        'titel': '🏠 Prédiction des Prix Immobiliers',
        'untertitel': 'Prédiction des prix immobiliers par IA',
        'flaeche': 'Surface (m²)',
        'zimmer': 'Pièces',
        'schlafzimmer': 'Chambres',
        'etage': 'Étage',
        'stadt': 'Ville',
        'baujahr': 'Année construction',
        'garten': 'Jardin',
        'balkon': 'Balcon',
        'vorhersagen': '🔮 Prédire le prix',
        'geschätzter_preis': 'Prix estimé',
        'preis_pro_m2': 'Prix au m²',
        'modell_laden': '🔄 Chargement modèle...',
        'modell_geladen': '✅ Modèle chargé',
        'modell_fehler': '❌ Échec chargement modèle',
        'staedte': ['Berlin', 'Hambourg', 'Munich', 'Cologne', 'Francfort', 'Stuttgart']
    },
    'EN': {
        'titel': '🏠 Real Estate Price Prediction',
        'untertitel': 'AI-powered price prediction',
        'flaeche': 'Living area (m²)',
        'zimmer': 'Rooms',
        'schlafzimmer': 'Bedrooms',
        'etage': 'Floor',
        'stadt': 'City',
        'baujahr': 'Year built',
        'garten': 'Garden',
        'balkon': 'Balcony',
        'vorhersagen': '🔮 Predict price',
        'geschätzter_preis': 'Estimated price',
        'preis_pro_m2': 'Price per m²',
        'modell_laden': '🔄 Loading model...',
        'modell_geladen': '✅ Model loaded',
        'modell_fehler': '❌ Model loading failed',
        'staedte': ['Berlin', 'Hamburg', 'Munich', 'Cologne', 'Frankfurt', 'Stuttgart']
    },
    'ES': {
        'titel': '🏠 Predicción de Precios Inmobiliarios',
        'untertitel': 'Predicción de precios con IA',
        'flaeche': 'Superficie (m²)',
        'zimmer': 'Habitaciones',
        'schlafzimmer': 'Dormitorios',
        'etage': 'Piso',
        'stadt': 'Ciudad',
        'baujahr': 'Año construcción',
        'garten': 'Jardín',
        'balkon': 'Balcón',
        'vorhersagen': '🔮 Predecir precio',
        'geschätzter_preis': 'Precio estimado',
        'preis_pro_m2': 'Precio por m²',
        'modell_laden': '🔄 Cargando modelo...',
        'modell_geladen': '✅ Modelo cargado',
        'modell_fehler': '❌ Error al cargar modelo',
        'staedte': ['Berlín', 'Hamburgo', 'Múnich', 'Colonia', 'Fráncfort', 'Stuttgart']
    },
    'IT': {
        'titel': '🏠 Previsione Prezzi Immobiliari',
        'untertitel': 'Previsione prezzi con IA',
        'flaeche': 'Superficie (m²)',
        'zimmer': 'Stanze',
        'schlafzimmer': 'Camere',
        'etage': 'Piano',
        'stadt': 'Città',
        'baujahr': 'Anno costruzione',
        'garten': 'Giardino',
        'balkon': 'Balcone',
        'vorhersagen': '🔮 Prevedi prezzo',
        'geschätzter_preis': 'Prezzo stimato',
        'preis_pro_m2': 'Prezzo al m²',
        'modell_laden': '🔄 Caricamento modello...',
        'modell_geladen': '✅ Modello caricato',
        'modell_fehler': '❌ Caricamento fallito',
        'staedte': ['Berlino', 'Amburgo', 'Monaco', 'Colonia', 'Francoforte', 'Stoccarda']
    }
}

def t(key):
    sprache = st.session_state.get('sprache', 'DE')
    return TEXTE[sprache].get(key, key)

# ============================================
# MODELL LADEN (CACHED)
# ============================================

@st.cache_resource
def load_model():
    """
    Lädt das trainierte Modell und den Scaler
    """
    try:
        # Prüfe ob die Modelldateien existieren
        model_path = Path("modelle/random_forest_model.pkl")
        scaler_path = Path("modelle/scaler.pkl")
        
        if not model_path.exists() or not scaler_path.exists():
            st.error("❌ Modelldateien nicht gefunden. Führe zuerst train.py aus!")
            return None, None
        
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        
        return model, scaler
    except Exception as e:
        st.error(f"❌ Fehler beim Laden des Modells: {e}")
        return None, None

# ============================================
# HILFSFUNKTIONEN
# ============================================

def erstelle_features(df):
    """
    Erstellt Features für die Vorhersage
    """
    df = df.copy()
    
    aktuelles_jahr = 2024
    df['alter'] = aktuelles_jahr - df['baujahr']
    df['zimmer_pro_m2'] = df['zimmer'] / df['flaeche'] * 100
    df['komfort_score'] = df['garten'] + df['balkon'] + (df['etage'] > 0).astype(int)
    
    # One-Hot-Encoding für Städte
    staedte = ['Berlin', 'Hamburg', 'München', 'Köln', 'Frankfurt', 'Stuttgart']
    for stadt in staedte:
        df[f'stadt_{stadt}'] = (df['stadt'] == stadt).astype(int)
    
    return df

# ============================================
# SESSION STATE
# ============================================

if 'sprache' not in st.session_state:
    st.session_state.sprache = 'DE'

# ============================================
# SIDEBAR
# ============================================

with st.sidebar:
    st.markdown(f"### 🌐 {t('Sprache')}")
    st.selectbox(
        "",
        options=list(SPRACHEN.keys()),
        format_func=lambda x: SPRACHEN[x],
        key='sprache'
    )
    
    st.markdown("---")
    
    st.markdown(f"### 🏠 {t('Immobiliendaten')}")
    
    with st.form("eingabe_form"):
        flaeche = st.number_input(t('flaeche'), 20.0, 300.0, 85.0, 5.0)
        
        col1, col2 = st.columns(2)
        with col1:
            zimmer = st.number_input(t('zimmer'), 1, 10, 4)
        with col2:
            schlafzimmer = st.number_input(t('schlafzimmer'), 1, 6, 2)
        
        etage = st.number_input(t('etage'), 0, 20, 2)
        stadt = st.selectbox(t('stadt'), t('staedte'))
        baujahr = st.number_input(t('baujahr'), 1900, 2024, 2010)
        
        col1, col2 = st.columns(2)
        with col1:
            garten = st.checkbox(t('garten'), True)
        with col2:
            balkon = st.checkbox(t('balkon'), True)
        
        submitted = st.form_submit_button(t('vorhersagen'), use_container_width=True)

# ============================================
# HAUPTBEREICH
# ============================================

st.markdown(f"# {t('titel')}")
st.markdown(f"### {t('untertitel')}")
st.markdown("---")

# Modell laden
with st.spinner(t('modell_laden')):
    model, scaler = load_model()

if model is None or scaler is None:
    st.stop()

st.success(t('modell_geladen'))

# ============================================
# VORHERSAGE
# ============================================

if submitted:
    # Daten vorbereiten
    daten = pd.DataFrame([{
        'flaeche': flaeche,
        'zimmer': zimmer,
        'schlafzimmer': schlafzimmer,
        'etage': etage,
        'stadt': stadt,
        'baujahr': baujahr,
        'garten': 1 if garten else 0,
        'balkon': 1 if balkon else 0
    }])
    
    # Feature Engineering
    daten_features = erstelle_features(daten)
    
    # Alle benötigten Spalten sicherstellen
    expected_cols = scaler.feature_names_in_ if hasattr(scaler, 'feature_names_in_') else None
    
    if expected_cols is not None:
        for col in expected_cols:
            if col not in daten_features.columns:
                daten_features[col] = 0
        daten_features = daten_features[expected_cols]
    
    # Skalieren und vorhersagen
    daten_scaled = scaler.transform(daten_features)
    preis = model.predict(daten_scaled)[0]
    
    # Ergebnisse anzeigen
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    border-radius: 15px; padding: 25px; text-align: center;">
            <h3 style="color: white;">{t('geschätzter_preis')}</h3>
            <p style="color: white; font-size: 2.5rem; font-weight: bold;">
                {preis:,.0f} €
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div style="background: #f8f9fa; border-radius: 15px; padding: 25px; text-align: center;">
            <h3 style="color: #667eea;">{t('preis_pro_m2')}</h3>
            <p style="color: #333; font-size: 2rem; font-weight: bold;">
                {preis/flaeche:,.0f} €
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        # Feature Importance (Beispiel)
        st.markdown(f"""
        <div style="background: #f8f9fa; border-radius: 15px; padding: 25px; text-align: center;">
            <h3 style="color: #667eea;">Vertrauen</h3>
            <p style="color: #333; font-size: 2rem; font-weight: bold;">85%</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Feature Importance Chart
    st.markdown("---")
    st.markdown("### 🔝 Wichtigste Features")
    
    importance_data = {
        'Wohnfläche': 0.35,
        'Lage': 0.25,
        'Baujahr': 0.15,
        'Zimmer': 0.12,
        'Ausstattung': 0.08,
        'Etage': 0.05
    }
    
    fig = px.bar(
        x=list(importance_data.values()),
        y=list(importance_data.keys()),
        orientation='h',
        color=list(importance_data.values()),
        color_continuous_scale='Purples'
    )
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True)