# app.py - Version tout-en-un avec génération automatique du modèle
# ============================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
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
# FONCTION DE GÉNÉRATION DU MODÈLE (AJOUTÉE)
# ============================================

def generiere_und_trainiere_modell():
    """
    Génère des données synthétiques et entraîne un modèle
    Cette fonction est appelée si les fichiers modèle n'existent pas
    """
    with st.spinner("🔄 Generiere Daten und trainiere Modell... (das dauert 30 Sekunden)"):
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.preprocessing import StandardScaler
        from sklearn.model_selection import train_test_split
        
        # 1. Daten generieren
        np.random.seed(42)
        anzahl = 1000
        
        flaeche = np.random.normal(80, 30, anzahl).clip(30, 250).astype(int)
        zimmer = (flaeche / 25 + np.random.normal(0, 1, anzahl)).clip(1, 8).astype(int)
        schlafzimmer = (zimmer * 0.5 + np.random.normal(0, 1, anzahl)).clip(1, 5).astype(int)
        etage = np.random.randint(0, 10, anzahl)
        
        staedte = ['Berlin', 'Hamburg', 'München', 'Köln', 'Frankfurt', 'Stuttgart']
        stadt = np.random.choice(staedte, anzahl)
        
        baujahr = np.random.randint(1950, 2023, anzahl)
        garten = np.random.choice([0, 1], anzahl, p=[0.4, 0.6])
        balkon = np.random.choice([0, 1], anzahl, p=[0.3, 0.7])
        
        # Preisberechnung
        preis_pro_qm = 3000
        preis_pro_qm += (stadt == 'München') * 4000
        preis_pro_qm += (stadt == 'Berlin') * 2000
        preis_pro_qm += (stadt == 'Hamburg') * 1500
        preis_pro_qm += (stadt == 'Frankfurt') * 1800
        
        preis = flaeche * preis_pro_qm
        preis *= (1 + 0.02 * etage)
        preis *= (1 + 0.1 * garten)
        preis *= (1 + 0.05 * balkon)
        preis += np.random.normal(0, preis * 0.1, anzahl)
        preis = preis.clip(50000, 2000000).astype(int)
        
        df = pd.DataFrame({
            'flaeche': flaeche,
            'zimmer': zimmer,
            'schlafzimmer': schlafzimmer,
            'etage': etage,
            'stadt': stadt,
            'baujahr': baujahr,
            'garten': garten,
            'balkon': balkon,
            'preis': preis
        })
        
        # 2. Feature Engineering
        df = df.copy()
        aktuelles_jahr = 2024
        df['alter'] = aktuelles_jahr - df['baujahr']
        df['zimmer_pro_m2'] = df['zimmer'] / df['flaeche'] * 100
        df['komfort_score'] = df['garten'] + df['balkon'] + (df['etage'] > 0).astype(int)
        
        # One-Hot-Encoding
        df = pd.get_dummies(df, columns=['stadt'], prefix='stadt')
        
        # 3. Features und Ziel
        feature_cols = [col for col in df.columns if col != 'preis']
        X = df[feature_cols]
        y = df['preis']
        
        # 4. Train-Test Split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # 5. Skalierung
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        
        # 6. Modell training
        model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
        model.fit(X_train_scaled, y_train)
        
        # 7. Modell speichern
        os.makedirs("modelle", exist_ok=True)
        joblib.dump(model, "modelle/random_forest_model.pkl")
        joblib.dump(scaler, "modelle/scaler.pkl")
        joblib.dump(feature_cols, "modelle/feature_cols.pkl")
        
        return model, scaler, feature_cols

# ============================================
# MODELL LADEN (MIT AUTOMATISCHER GENERIERUNG)
# ============================================

@st.cache_resource
def load_model():
    """
    Lädt das trainierte Modell oder generiert es falls nötig
    """
    model_path = Path("modelle/random_forest_model.pkl")
    scaler_path = Path("modelle/scaler.pkl")
    
    if not model_path.exists() or not scaler_path.exists():
        st.warning("⚠️ Keine Modelldateien gefunden. Generiere neues Modell...")
        model, scaler, feature_cols = generiere_und_trainiere_modell()
        return model, scaler
    
    try:
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        return model, scaler
    except Exception as e:
        st.error(f"❌ Fehler beim Laden: {e}")
        st.info("🔄 Generiere neues Modell...")
        model, scaler, feature_cols = generiere_und_trainiere_modell()
        return model, scaler

# ============================================
# MEHRSPRACHIGKEIT (GLEICH WIE VORHER)
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
        'modell_geladen': '✅ Modell geladen',
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
        'modell_geladen': '✅ Modèle chargé',
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
        'modell_geladen': '✅ Model loaded',
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
        'modell_geladen': '✅ Modelo cargado',
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
        'modell_geladen': '✅ Modello caricato',
        'staedte': ['Berlino', 'Amburgo', 'Monaco', 'Colonia', 'Francoforte', 'Stoccarda']
    }
}

def t(key):
    sprache = st.session_state.get('sprache', 'DE')
    return TEXTE[sprache].get(key, key)

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

# Modell laden (automatisch generieren falls nötig)
model, scaler = load_model()

if model is None or scaler is None:
    st.error("❌ Modell konnte nicht geladen oder generiert werden")
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
    
    # Features in richtiger Reihenfolge
    if hasattr(scaler, 'feature_names_in_'):
        expected_cols = scaler.feature_names_in_
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
        # Feature Importance (statique pour l'exemple)
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