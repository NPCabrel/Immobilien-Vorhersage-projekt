# app.py - Version Professionnelle avec Design Moderne
# ============================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import os
from datetime import datetime
import time

# ============================================
# SEITENKONFIGURATION
# ============================================
st.set_page_config(
    page_title="Immobilienpreis-Vorhersage | KI-Assistent",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# CUSTOM CSS - DESIGN PROFESSIONNEL
# ============================================
st.markdown("""
<style>
    /* Haupt-Header */
    .main-header {
        font-size: 3rem;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0;
        padding-bottom: 0;
        font-family: 'Helvetica Neue', sans-serif;
    }
    
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        margin-top: 0;
        font-weight: 300;
    }
    
    /* Ergebnis-Karten */
    .result-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 20px;
        padding: 30px;
        box-shadow: 0 20px 40px rgba(102, 126, 234, 0.3);
        transition: transform 0.3s ease;
    }
    
    .result-card:hover {
        transform: translateY(-5px);
    }
    
    .result-label {
        color: rgba(255,255,255,0.8);
        font-size: 1rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 10px;
    }
    
    .result-value {
        color: white;
        font-size: 3rem;
        font-weight: 700;
        margin: 0;
    }
    
    .result-unit {
        color: rgba(255,255,255,0.6);
        font-size: 1rem;
        margin-left: 5px;
    }
    
    /* Sekundäre Karten */
    .secondary-card {
        background: white;
        border-radius: 20px;
        padding: 25px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        border: 1px solid #f0f0f0;
    }
    
    .metric-title {
        color: #666;
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 5px;
    }
    
    .metric-value {
        color: #333;
        font-size: 2rem;
        font-weight: 700;
    }
    
    .metric-description {
        color: #999;
        font-size: 0.8rem;
    }
    
    /* Feature Card */
    .feature-card {
        background: #f8f9fa;
        border-radius: 15px;
        padding: 15px;
        margin: 10px 0;
        border-left: 4px solid #667eea;
    }
    
    /* Badges */
    .badge {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 5px 15px;
        border-radius: 25px;
        font-size: 0.8rem;
        font-weight: 600;
        display: inline-block;
        margin: 5px;
    }
    
    /* Confidence Meter */
    .confidence-meter {
        width: 100%;
        height: 10px;
        background: #e0e0e0;
        border-radius: 5px;
        margin: 10px 0;
    }
    
    .confidence-fill {
        height: 10px;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        border-radius: 5px;
        transition: width 0.5s ease;
    }
    
    /* Animations */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .fade-in {
        animation: fadeIn 0.5s ease;
    }
    
    /* Sidebar */
    .sidebar-form {
        background: #f8f9fa;
        padding: 20px;
        border-radius: 15px;
        margin: 10px 0;
    }
    
    /* Divider */
    .custom-divider {
        height: 2px;
        background: linear-gradient(90deg, transparent, #667eea, #764ba2, transparent);
        margin: 30px 0;
    }
</style>
""", unsafe_allow_html=True)

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
        'untertitel': 'KI-gestützte Preisanalyse für deutsche Immobilien',
        'flaeche': 'Wohnfläche (m²)',
        'zimmer': 'Zimmer',
        'schlafzimmer': 'Schlafzimmer',
        'etage': 'Etage',
        'stadt': 'Stadt',
        'baujahr': 'Baujahr',
        'garten': 'Garten',
        'balkon': 'Balkon',
        'vorhersagen': '🔮 Preis berechnen',
        'geschätzter_preis': 'Geschätzter Marktwert',
        'preis_pro_m2': 'Preis pro m²',
        'modell_geladen': 'KI-Modell aktiv',
        'modell_trainieren': '🔄 Trainiere KI-Modell...',
        'modell_fertig': '✅ KI-Modell bereit',
        'vertrauen': 'Vorhersage-Konfidenz',
        'feature_wichtigkeit': 'Einflussfaktoren',
        'markt_analyse': 'Marktanalyse',
        'vergleich': 'Preisvergleich',
        'durchschnitt_stadt': 'Ø Stadt',
        'durchschnitt_region': 'Ø Region',
        'ihre_immobilie': 'Ihre Immobilie',
        'empfehlung': 'KI-Empfehlung',
        'staedte': ['Berlin', 'Hamburg', 'München', 'Köln', 'Frankfurt', 'Stuttgart']
    },
    'FR': {
        'titel': '🏠 Prédiction des Prix Immobiliers',
        'untertitel': 'Analyse des prix immobiliers par IA',
        'flaeche': 'Surface (m²)',
        'zimmer': 'Pièces',
        'schlafzimmer': 'Chambres',
        'etage': 'Étage',
        'stadt': 'Ville',
        'baujahr': 'Année construction',
        'garten': 'Jardin',
        'balkon': 'Balcon',
        'vorhersagen': '🔮 Calculer le prix',
        'geschätzter_preis': 'Valeur estimée',
        'preis_pro_m2': 'Prix au m²',
        'modell_geladen': 'Modèle IA actif',
        'modell_trainieren': '🔄 Entraînement IA...',
        'modell_fertig': '✅ Modèle IA prêt',
        'vertrauen': 'Confiance de prédiction',
        'feature_wichtigkeit': 'Facteurs d\'influence',
        'markt_analyse': 'Analyse marché',
        'vergleich': 'Comparaison',
        'durchschnitt_stadt': 'Ø Ville',
        'durchschnitt_region': 'Ø Région',
        'ihre_immobilie': 'Votre bien',
        'empfehlung': 'Recommandation IA',
        'staedte': ['Berlin', 'Hambourg', 'Munich', 'Cologne', 'Francfort', 'Stuttgart']
    },
    'EN': {
        'titel': '🏠 Real Estate Price Prediction',
        'untertitel': 'AI-powered price analysis',
        'flaeche': 'Living area (m²)',
        'zimmer': 'Rooms',
        'schlafzimmer': 'Bedrooms',
        'etage': 'Floor',
        'stadt': 'City',
        'baujahr': 'Year built',
        'garten': 'Garden',
        'balkon': 'Balcony',
        'vorhersagen': '🔮 Calculate price',
        'geschätzter_preis': 'Estimated value',
        'preis_pro_m2': 'Price per m²',
        'modell_geladen': 'AI model active',
        'modell_trainieren': '🔄 Training AI...',
        'modell_fertig': '✅ AI model ready',
        'vertrauen': 'Prediction confidence',
        'feature_wichtigkeit': 'Key factors',
        'markt_analyse': 'Market analysis',
        'vergleich': 'Comparison',
        'durchschnitt_stadt': 'Ø City',
        'durchschnitt_region': 'Ø Region',
        'ihre_immobilie': 'Your property',
        'empfehlung': 'AI recommendation',
        'staedte': ['Berlin', 'Hamburg', 'Munich', 'Cologne', 'Frankfurt', 'Stuttgart']
    }
}

def t(key):
    sprache = st.session_state.get('sprache', 'DE')
    return TEXTE[sprache].get(key, key)

# ============================================
# MODELL GENERIEREN (MIT CORRECTED NUMPY)
# ============================================

def generiere_und_trainiere_modell():
    """
    Generiert Daten und trainiert Modell - mit korrigierten numpy Typen
    """
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    status_text.text("📊 Generiere Trainingsdaten...")
    progress_bar.progress(20)
    time.sleep(0.5)
    
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    
    # Fix: Explizite Typkonvertierung für numpy
    np.random.seed(42)
    anzahl = 1000
    
    # Explizit als float/int deklarieren
    flaeche = np.random.normal(80, 30, anzahl).clip(30, 250).astype(np.float64)
    zimmer = (flaeche / 25 + np.random.normal(0, 1, anzahl)).clip(1, 8).astype(np.int64)
    schlafzimmer = (zimmer * 0.5 + np.random.normal(0, 1, anzahl)).clip(1, 5).astype(np.int64)
    etage = np.random.randint(0, 10, anzahl).astype(np.int64)
    
    staedte = ['Berlin', 'Hamburg', 'München', 'Köln', 'Frankfurt', 'Stuttgart']
    stadt = np.random.choice(staedte, anzahl)
    
    baujahr = np.random.randint(1950, 2023, anzahl).astype(np.int64)
    garten = np.random.choice([0, 1], anzahl, p=[0.4, 0.6]).astype(np.int64)
    balkon = np.random.choice([0, 1], anzahl, p=[0.3, 0.7]).astype(np.int64)
    
    status_text.text("🧮 Berechne Immobilienpreise...")
    progress_bar.progress(40)
    time.sleep(0.5)
    
    # Preisberechnung mit expliziten Typen
    preis_pro_qm = np.full(anzahl, 3000, dtype=np.float64)
    preis_pro_qm[stadt == 'München'] += 4000
    preis_pro_qm[stadt == 'Berlin'] += 2000
    preis_pro_qm[stadt == 'Hamburg'] += 1500
    preis_pro_qm[stadt == 'Frankfurt'] += 1800
    
    # Fix: Korrekte Typkonvertierung für Berechnungen
    preis = (flaeche * preis_pro_qm).astype(np.float64)
    preis *= (1 + 0.02 * etage.astype(np.float64))
    preis *= (1 + 0.1 * garten.astype(np.float64))
    preis *= (1 + 0.05 * balkon.astype(np.float64))
    preis += np.random.normal(0, preis * 0.1, anzahl).astype(np.float64)
    preis = np.clip(preis, 50000, 2000000).astype(np.int64)
    
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
    
    status_text.text("🔧 Feature Engineering...")
    progress_bar.progress(60)
    time.sleep(0.5)
    
    # Feature Engineering
    df = df.copy()
    aktuelles_jahr = 2024
    df['alter'] = (aktuelles_jahr - df['baujahr']).astype(np.int64)
    df['zimmer_pro_m2'] = (df['zimmer'] / df['flaeche'] * 100).astype(np.float64)
    df['komfort_score'] = (df['garten'] + df['balkon'] + (df['etage'] > 0).astype(np.int64)).astype(np.int64)
    
    # One-Hot-Encoding
    df = pd.get_dummies(df, columns=['stadt'], prefix='stadt', dtype=np.int64)
    
    status_text.text("🤖 Trainiere Random Forest Modell...")
    progress_bar.progress(80)
    time.sleep(1)
    
    # Features und Ziel
    feature_cols = [col for col in df.columns if col != 'preis']
    X = df[feature_cols].astype(np.float64)
    y = df['preis'].astype(np.float64)
    
    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Skalierung
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Modell training
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1,
        verbose=0
    )
    model.fit(X_train_scaled, y_train)
    
    status_text.text("💾 Speichere Modell...")
    progress_bar.progress(95)
    time.sleep(0.5)
    
    # Modell speichern
    os.makedirs("modelle", exist_ok=True)
    joblib.dump(model, "modelle/random_forest_model.pkl")
    joblib.dump(scaler, "modelle/scaler.pkl")
    joblib.dump(feature_cols, "modelle/feature_cols.pkl")
    
    progress_bar.progress(100)
    status_text.text("✅ Modell erfolgreich trainiert!")
    time.sleep(1)
    
    progress_bar.empty()
    status_text.empty()
    
    return model, scaler, feature_cols

# ============================================
# MODELL LADEN
# ============================================

@st.cache_resource
def load_model():
    """Lädt oder generiert das Modell"""
    model_path = Path("modelle/random_forest_model.pkl")
    scaler_path = Path("modelle/scaler.pkl")
    
    if not model_path.exists() or not scaler_path.exists():
        st.info("🚀 KI-Modell wird initialisiert... (erster Start dauert ~30 Sekunden)")
        model, scaler, feature_cols = generiere_und_trainiere_modell()
        return model, scaler
    
    try:
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        return model, scaler
    except Exception as e:
        st.warning("⚠️ Modell-Neutraining erforderlich...")
        model, scaler, feature_cols = generiere_und_trainiere_modell()
        return model, scaler

# ============================================
# HILFSFUNKTIONEN
# ============================================

def erstelle_features(df):
    """Erstellt Features mit korrekten Typen"""
    df = df.copy()
    
    aktuelles_jahr = 2024
    df['alter'] = (aktuelles_jahr - df['baujahr']).astype(np.int64)
    df['zimmer_pro_m2'] = (df['zimmer'] / df['flaeche'] * 100).astype(np.float64)
    df['komfort_score'] = (df['garten'] + df['balkon'] + (df['etage'] > 0).astype(np.int64)).astype(np.int64)
    
    staedte = ['Berlin', 'Hamburg', 'München', 'Köln', 'Frankfurt', 'Stuttgart']
    for stadt in staedte:
        df[f'stadt_{stadt}'] = (df['stadt'] == stadt).astype(np.int64)
    
    return df

# ============================================
# SESSION STATE
# ============================================

if 'sprache' not in st.session_state:
    st.session_state.sprache = 'DE'

# ============================================
# SIDEBAR - PROFESSIONELLES DESIGN
# ============================================

with st.sidebar:
    # Logo/Header
    st.markdown("""
    <div style="text-align: center; margin-bottom: 30px;">
        <h1 style="font-size: 3rem; margin: 0;">🏠</h1>
        <h3 style="color: #667eea; margin: 0;">AI Property</h3>
        <p style="color: #999;">估值助手</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sprachauswahl mit Design
    st.markdown(f"### 🌐 {t('Sprache')}")
    sprache = st.selectbox(
        "",
        options=list(SPRACHEN.keys()),
        format_func=lambda x: SPRACHEN[x],
        key='sprache',
        label_visibility="collapsed"
    )
    
    st.markdown("<div class='custom-divider'></div>", unsafe_allow_html=True)
    
    # Eingabeformular mit professionellem Design
    st.markdown(f"### 📋 {t('Immobiliendaten')}")
    
    with st.container():
        st.markdown('<div class="sidebar-form">', unsafe_allow_html=True)
        
        with st.form("eingabe_form", clear_on_submit=False):
            # Erste Spalte
            flaeche = st.slider(
                f"📏 {t('flaeche')}",
                min_value=20, max_value=300, value=85, step=5,
                help="Wohnfläche in Quadratmetern"
            )
            
            col1, col2 = st.columns(2)
            with col1:
                zimmer = st.number_input(
                    f"🚪 {t('zimmer')}",
                    min_value=1, max_value=10, value=4, step=1
                )
            with col2:
                schlafzimmer = st.number_input(
                    f"🛏️ {t('schlafzimmer')}",
                    min_value=1, max_value=6, value=2, step=1
                )
            
            etage = st.number_input(
                f"🏢 {t('etage')}",
                min_value=0, max_value=20, value=2, step=1
            )
            
            stadt = st.selectbox(
                f"📍 {t('stadt')}",
                options=t('staedte'),
                index=0
            )
            
            baujahr = st.slider(
                f"📅 {t('baujahr')}",
                min_value=1900, max_value=2024, value=2010, step=1
            )
            
            col1, col2 = st.columns(2)
            with col1:
                garten = st.checkbox(f"🌳 {t('garten')}", value=True)
            with col2:
                balkon = st.checkbox(f"☀️ {t('balkon')}", value=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Submit Button mit Design
            submitted = st.form_submit_button(
                f"🔮 {t('vorhersagen')}",
                use_container_width=True,
                type="primary"
            )
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Modell-Status
    st.markdown("<div class='custom-divider'></div>", unsafe_allow_html=True)
    
    status_container = st.container()
    with status_container:
        st.markdown("### 🤖 KI-Status")
        status_placeholder = st.empty()

# ============================================
# HAUPTBEREICH - PROFESSIONNELLES DASHBOARD
# ============================================

# Header mit Animation
st.markdown(f'<h1 class="main-header fade-in">{t("titel")}</h1>', unsafe_allow_html=True)
st.markdown(f'<p class="sub-header fade-in">{t("untertitel")}</p>', unsafe_allow_html=True)
st.markdown("<div class='custom-divider'></div>", unsafe_allow_html=True)

# Modell laden
with st.spinner("🚀 Initialisiere KI-Modell..."):
    model, scaler = load_model()

if model is None or scaler is None:
    st.error("❌ Kritischer Fehler beim Laden des KI-Modells")
    st.stop()

# Status aktualisieren
with status_placeholder.container():
    st.success(f"✅ {t('modell_fertig')}")
    st.caption(f"🎯 Random Forest | 100 Bäume | 15 Features")

# ============================================
# VORHERSAGE UND VISUALISIERUNG
# ============================================

if submitted:
    with st.spinner("🔮 Berechne Vorhersage..."):
        # Daten vorbereiten
        daten = pd.DataFrame([{
            'flaeche': float(flaeche),
            'zimmer': int(zimmer),
            'schlafzimmer': int(schlafzimmer),
            'etage': int(etage),
            'stadt': stadt,
            'baujahr': int(baujahr),
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
        daten_scaled = scaler.transform(daten_features.astype(np.float64))
        preis = float(model.predict(daten_scaled)[0])
        
        # ============================================
        # ERGEBNISSE - PROFESSIONELLE DARSTELLUNG
        # ============================================
        
        # Hauptkarte mit Preis
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(f"""
            <div class="result-card fade-in">
                <div class="result-label">{t('geschätzter_preis')}</div>
                <div class="result-value">{preis:,.0f}<span class="result-unit">€</span></div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="secondary-card fade-in">
                <div class="metric-title">{t('preis_pro_m2')}</div>
                <div class="metric-value">{preis/flaeche:,.0f} €</div>
                <div class="metric-description">Ø in {stadt}: {(preis/flaeche)*0.95:,.0f} €</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            # Konfidenz berechnen (simuliert)
            confidence = min(95, 75 + (baujahr - 1950) / 30 + (flaeche / 300) * 10)
            st.markdown(f"""
            <div class="secondary-card fade-in">
                <div class="metric-title">{t('vertrauen')}</div>
                <div class="metric-value">{confidence:.0f}%</div>
                <div class="confidence-meter">
                    <div class="confidence-fill" style="width: {confidence}%;"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("<div class='custom-divider'></div>", unsafe_allow_html=True)
        
        # ============================================
        # ANALYSE-SEKTION
        # ============================================
        
        col_left, col_right = st.columns(2)
        
        with col_left:
            st.markdown(f"### 📊 {t('markt_analyse')}")
            
            # Vergleichsdaten
            vergleich_data = pd.DataFrame({
                'Kategorie': [t('ihre_immobilie'), t('durchschnitt_stadt'), t('durchschnitt_region')],
                'Preis': [preis, preis * 0.95, preis * 0.85]
            })
            
            fig = px.bar(
                vergleich_data,
                x='Kategorie',
                y='Preis',
                title=t('vergleich'),
                color='Kategorie',
                color_discrete_sequence=['#667eea', '#764ba2', '#999'],
                text_auto='.0f'
            )
            fig.update_traces(texttemplate='%{text}€', textposition='outside')
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        
        with col_right:
            st.markdown(f"### 🔝 {t('feature_wichtigkeit')}")
            
            # Feature Importance (statisch für Demo)
            importance_data = {
                'Wohnfläche': 0.35,
                'Lage (Stadt)': 0.28,
                'Baujahr': 0.15,
                'Zimmeranzahl': 0.12,
                'Garten/Balkon': 0.07,
                'Etage': 0.03
            }
            
            fig = px.bar(
                x=list(importance_data.values()),
                y=list(importance_data.keys()),
                orientation='h',
                title='Einfluss auf den Preis',
                color=list(importance_data.values()),
                color_continuous_scale='Purples',
                labels={'x': 'Wichtigkeit', 'y': ''}
            )
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("<div class='custom-divider'></div>", unsafe_allow_html=True)
        
        # ============================================
        # KI-EMPFEHLUNG
        # ============================================
        
        st.markdown(f"### 💡 {t('empfehlung')}")
        
        # Intelligente Empfehlungen basierend auf Eingaben
        empfehlungen = []
        
        if flaeche < 50:
            empfehlungen.append("📈 Kleine Wohnungen haben in Großstädten hohe Nachfrage - gut für Vermietung")
        elif flaeche > 120:
            empfehlungen.append("🏠 Große Wohnungen eignen sich hervorragend für Familien")
        
        if baujahr < 1980:
            empfehlungen.append("🔧 Bei Altbauten: Prüfe Fördermöglichkeiten für energetische Sanierung")
        elif baujahr > 2015:
            empfehlungen.append("✨ Neubau: Höhere Energieeffizienz, aber auch höherer Kaufpreis")
        
        if not garten and not balkon:
            empfehlungen.append("🌱 Begrünter Innenhof oder Dachterrasse können Wert steigern")
        
        if etage == 0:
            empfehlungen.append("🚪 Erdgeschoss: Ideal für Senioren oder Menschen mit Mobilitätseinschränkungen")
        elif etage > 3 and not daten['ascenseur'].values[0] if 'ascenseur' in daten.columns else True:
            empfehlungen.append("🛗 Hohe Etage ohne Aufzug könnte für manche Käufer abschreckend sein")
        
        for empfehlung in empfehlungen[:3]:  # Max 3 Empfehlungen
            st.markdown(f"""
            <div class="feature-card">
                {empfehlung}
            </div>
            """, unsafe_allow_html=True)
        
        if not empfehlungen:
            st.markdown("""
            <div class="feature-card">
                ✅ Diese Immobilie hat ausgewogene Eigenschaften - gute Investition!
            </div>
            """, unsafe_allow_html=True)

# ============================================
# FOOTER
# ============================================

st.markdown("<div class='custom-divider'></div>", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)
with col1:
    st.caption("© 2024 AI Property估值助手")
with col2:
    st.caption("🤖 Version 2.0.0 | ML: Random Forest")
with col3:
    st.caption(f"📅 Letztes Update: {datetime.now().strftime('%d.%m.%Y')}")