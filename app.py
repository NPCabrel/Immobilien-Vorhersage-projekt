# app.py
# ============================================
# STREAMLIT FRONTEND FÜR IMMOBILIENPREISVORHERSAGE
# MEHRSPRACHIG (DE/FR/EN/ES/IT) - STANDARD DEUTSCH
# ============================================

import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from datetime import datetime

# ============================================
# SEITENKONFIGURATION
# ============================================
st.set_page_config(
    page_title="Immobilienpreis-Vorhersage",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# MEHRSPRACHIGKEIT (DEUTSCH STANDARD)
# ============================================

# Verfügbare Sprachen
SPRACHEN = {
    'DE': 'Deutsch',
    'FR': 'Français',
    'EN': 'English',
    'ES': 'Español',
    'IT': 'Italiano'
}

# Übersetzungen
TEXTE = {
    'DE': {
        # Allgemein
        'titel': '🏠 Immobilienpreis-Vorhersage',
        'untertitel': 'KI-gestützte Vorhersage für Immobilienpreise in Deutschland',
        'api_status': 'API Status',
        'api_online': '✅ API ist online',
        'api_offline': '❌ API nicht erreichbar',
        'modell_geladen': 'Modell geladen',
        'features': 'Features',
        
        # Sidebar
        'sprache': 'Sprache',
        'immobilien_daten': '🏠 Immobiliendaten eingeben',
        'flaeche': 'Wohnfläche (m²)',
        'zimmer': 'Zimmer',
        'schlafzimmer': 'Schlafzimmer',
        'etage': 'Etage',
        'stadt': 'Stadt',
        'plz': 'Postleitzahl',
        'baujahr': 'Baujahr',
        'entfernung_bahn': 'Entfernung zum Bahnhof (km)',
        'garten': 'Garten',
        'balkon': 'Balkon',
        'vorhersagen': '🔮 Preis vorhersagen',
        
        # Ergebnisse
        'ergebnisse': '📊 Vorhersage-Ergebnisse',
        'geschätzter_preis': 'Geschätzter Immobilienpreis',
        'preis_pro_m2': 'Preis pro m²',
        'intervall': 'Vorhersageintervall',
        'verwendetes_modell': 'Verwendetes Modell',
        'feature_wichtigkeit': '🔝 Feature-Wichtigkeit',
        
        # Info
        'informationen': 'ℹ️ Informationen',
        'über_das_modell': 'Über das Modell',
        'modell_beschreibung': 'Dieses Modell wurde mit einem Random Forest Algorithmus trainiert, um Immobilienpreise in deutschen Städten vorherzusagen.',
        'feature_beschreibung': 'Die Vorhersage basiert auf folgenden Merkmalen: Wohnfläche, Zimmeranzahl, Lage, Baujahr, Ausstattung.',
        'genauigkeit': 'Genauigkeit: R² = 0.85',
        
        # Buttons
        'zurücksetzen': '🔄 Zurücksetzen',
        'beispiel_laden': '📋 Beispiel laden',
        
        # Städte
        'staedte': ['Berlin', 'Hamburg', 'München', 'Köln', 'Frankfurt', 'Stuttgart'],
        
        # Ja/Nein
        'ja': 'Ja',
        'nein': 'Nein',
    },
    'FR': {
        'titel': '🏠 Prédiction des Prix Immobiliers',
        'untertitel': 'Prédiction des prix immobiliers en Allemagne par IA',
        'api_status': 'Statut API',
        'api_online': '✅ API en ligne',
        'api_offline': '❌ API inaccessible',
        'modell_geladen': 'Modèle chargé',
        'features': 'Caractéristiques',
        'sprache': 'Langue',
        'immobilien_daten': '🏠 Données du bien',
        'flaeche': 'Surface (m²)',
        'zimmer': 'Pièces',
        'schlafzimmer': 'Chambres',
        'etage': 'Étage',
        'stadt': 'Ville',
        'plz': 'Code postal',
        'baujahr': 'Année construction',
        'entfernung_bahn': 'Distance gare (km)',
        'garten': 'Jardin',
        'balkon': 'Balcon',
        'vorhersagen': '🔮 Prédire le prix',
        'ergebnisse': '📊 Résultats',
        'geschätzter_preis': 'Prix estimé',
        'preis_pro_m2': 'Prix au m²',
        'intervall': 'Intervalle de prédiction',
        'verwendetes_modell': 'Modèle utilisé',
        'feature_wichtigkeit': '🔝 Importance des caractéristiques',
        'informationen': 'ℹ️ Informations',
        'über_das_modell': 'À propos du modèle',
        'modell_beschreibung': 'Modèle Random Forest pour prédire les prix immobiliers en Allemagne.',
        'feature_beschreibung': 'Prédiction basée sur: surface, pièces, localisation, année, équipements.',
        'genauigkeit': 'Précision: R² = 0.85',
        'zurücksetzen': '🔄 Réinitialiser',
        'beispiel_laden': '📋 Exemple',
        'staedte': ['Berlin', 'Hambourg', 'Munich', 'Cologne', 'Francfort', 'Stuttgart'],
        'ja': 'Oui',
        'nein': 'Non',
    },
    'EN': {
        'titel': '🏠 Real Estate Price Prediction',
        'untertitel': 'AI-powered price prediction for German real estate',
        'api_status': 'API Status',
        'api_online': '✅ API online',
        'api_offline': '❌ API unavailable',
        'modell_geladen': 'Model loaded',
        'features': 'Features',
        'sprache': 'Language',
        'immobilien_daten': '🏠 Property Data',
        'flaeche': 'Living area (m²)',
        'zimmer': 'Rooms',
        'schlafzimmer': 'Bedrooms',
        'etage': 'Floor',
        'stadt': 'City',
        'plz': 'Postal code',
        'baujahr': 'Year built',
        'entfernung_bahn': 'Distance to station (km)',
        'garten': 'Garden',
        'balkon': 'Balcony',
        'vorhersagen': '🔮 Predict price',
        'ergebnisse': '📊 Results',
        'geschätzter_preis': 'Estimated price',
        'preis_pro_m2': 'Price per m²',
        'intervall': 'Prediction interval',
        'verwendetes_modell': 'Model used',
        'feature_wichtigkeit': '🔝 Feature Importance',
        'informationen': 'ℹ️ Information',
        'über_das_modell': 'About the model',
        'modell_beschreibung': 'Random Forest model trained to predict real estate prices in German cities.',
        'feature_beschreibung': 'Prediction based on: area, rooms, location, year, amenities.',
        'genauigkeit': 'Accuracy: R² = 0.85',
        'zurücksetzen': '🔄 Reset',
        'beispiel_laden': '📋 Load example',
        'staedte': ['Berlin', 'Hamburg', 'Munich', 'Cologne', 'Frankfurt', 'Stuttgart'],
        'ja': 'Yes',
        'nein': 'No',
    },
    'ES': {
        'titel': '🏠 Predicción de Precios Inmobiliarios',
        'untertitel': 'Predicción de precios inmobiliarios en Alemania con IA',
        'api_status': 'Estado API',
        'api_online': '✅ API en línea',
        'api_offline': '❌ API no disponible',
        'modell_geladen': 'Modelo cargado',
        'features': 'Características',
        'sprache': 'Idioma',
        'immobilien_daten': '🏠 Datos de la propiedad',
        'flaeche': 'Superficie (m²)',
        'zimmer': 'Habitaciones',
        'schlafzimmer': 'Dormitorios',
        'etage': 'Piso',
        'stadt': 'Ciudad',
        'plz': 'Código postal',
        'baujahr': 'Año construcción',
        'entfernung_bahn': 'Distancia estación (km)',
        'garten': 'Jardín',
        'balkon': 'Balcón',
        'vorhersagen': '🔮 Predecir precio',
        'ergebnisse': '📊 Resultados',
        'geschätzter_preis': 'Precio estimado',
        'preis_pro_m2': 'Precio por m²',
        'intervall': 'Intervalo de predicción',
        'verwendetes_modell': 'Modelo utilizado',
        'feature_wichtigkeit': '🔝 Importancia de características',
        'informationen': 'ℹ️ Información',
        'über_das_modell': 'Sobre el modelo',
        'modell_beschreibung': 'Modelo Random Forest para predecir precios inmobiliarios en Alemania.',
        'feature_beschreibung': 'Predicción basada en: superficie, habitaciones, ubicación, año, equipamiento.',
        'genauigkeit': 'Precisión: R² = 0.85',
        'zurücksetzen': '🔄 Reiniciar',
        'beispiel_laden': '📋 Cargar ejemplo',
        'staedte': ['Berlín', 'Hamburgo', 'Múnich', 'Colonia', 'Fráncfort', 'Stuttgart'],
        'ja': 'Sí',
        'nein': 'No',
    },
    'IT': {
        'titel': '🏠 Previsione Prezzi Immobiliari',
        'untertitel': 'Previsione prezzi immobiliari in Germania con IA',
        'api_status': 'Stato API',
        'api_online': '✅ API online',
        'api_offline': '❌ API non disponibile',
        'modell_geladen': 'Modello caricato',
        'features': 'Caratteristiche',
        'sprache': 'Lingua',
        'immobilien_daten': '🏠 Dati immobile',
        'flaeche': 'Superficie (m²)',
        'zimmer': 'Stanze',
        'schlafzimmer': 'Camere da letto',
        'etage': 'Piano',
        'stadt': 'Città',
        'plz': 'Codice postale',
        'baujahr': 'Anno costruzione',
        'entfernung_bahn': 'Distanza stazione (km)',
        'garten': 'Giardino',
        'balkon': 'Balcone',
        'vorhersagen': '🔮 Prevedi prezzo',
        'ergebnisse': '📊 Risultati',
        'geschätzter_preis': 'Prezzo stimato',
        'preis_pro_m2': 'Prezzo al m²',
        'intervall': 'Intervallo previsione',
        'verwendetes_modell': 'Modello utilizzato',
        'feature_wichtigkeit': '🔝 Importanza caratteristiche',
        'informationen': 'ℹ️ Informazioni',
        'über_das_modell': 'Sul modello',
        'modell_beschreibung': 'Modello Random Forest per prevedere prezzi immobiliari in Germania.',
        'feature_beschreibung': 'Previsione basata su: superficie, stanze, posizione, anno, servizi.',
        'genauigkeit': 'Precisione: R² = 0.85',
        'zurücksetzen': '🔄 Reimposta',
        'beispiel_laden': '📋 Carica esempio',
        'staedte': ['Berlino', 'Amburgo', 'Monaco', 'Colonia', 'Francoforte', 'Stoccarda'],
        'ja': 'Sì',
        'nein': 'No',
    }
}

# ============================================
# HILFSFUNKTIONEN
# ============================================

def t(schluessel):
    """Gibt den übersetzten Text zurück"""
    sprache = st.session_state.get('sprache', 'DE')
    return TEXTE[sprache].get(schluessel, schluessel)

def check_api():
    """Prüft ob die API erreichbar ist"""
    try:
        response = requests.get("http://localhost:8000/health", timeout=2)
        return response.status_code == 200
    except:
        return False

def get_feature_importance():
    """Erstellt Beispiel-Feature-Importance für die Visualisierung"""
    # In einer echten App würde das aus dem Modell kommen
    return {
        'Wohnfläche': 0.35,
        'Lage (Stadt)': 0.25,
        'Baujahr': 0.15,
        'Zimmeranzahl': 0.12,
        'Garten/Balkon': 0.08,
        'Etage': 0.05
    }

# ============================================
# SESSION STATE INITIALISIERUNG
# ============================================

if 'sprache' not in st.session_state:
    st.session_state.sprache = 'DE'  # Standard: Deutsch

# ============================================
# SIDEBAR - SPRACHAUSWAHL UND EINGABEFORMULAR
# ============================================

with st.sidebar:
    # Sprachauswahl
    st.markdown(f"### 🌐 {t('sprache')}")
    sprache = st.selectbox(
        "",
        options=list(SPRACHEN.keys()),
        format_func=lambda x: SPRACHEN[x],
        key='sprache'
    )
    
    st.markdown("---")
    
    # API Status
    st.markdown(f"### {t('api_status')}")
    api_online = check_api()
    if api_online:
        st.success(t('api_online'))
    else:
        st.error(t('api_offline'))
        st.warning("Starte zuerst die API mit: python api.py")
    
    st.markdown("---")
    
    # Eingabeformular
    st.markdown(f"### {t('immobilien_daten')}")
    
    with st.form("eingabe_formular"):
        flaeche = st.number_input(
            t('flaeche'),
            min_value=20.0,
            max_value=300.0,
            value=85.0,
            step=5.0
        )
        
        col1, col2 = st.columns(2)
        with col1:
            zimmer = st.number_input(
                t('zimmer'),
                min_value=1,
                max_value=10,
                value=4,
                step=1
            )
        with col2:
            schlafzimmer = st.number_input(
                t('schlafzimmer'),
                min_value=1,
                max_value=6,
                value=2,
                step=1
            )
        
        etage = st.number_input(
            t('etage'),
            min_value=0,
            max_value=20,
            value=2,
            step=1
        )
        
        stadt = st.selectbox(
            t('stadt'),
            options=t('staedte')
        )
        
        plz = st.number_input(
            t('plz'),
            min_value=10000,
            max_value=99999,
            value=10115,
            step=100
        )
        
        baujahr = st.number_input(
            t('baujahr'),
            min_value=1900,
            max_value=2024,
            value=2010,
            step=1
        )
        
        entfernung_bahn = st.slider(
            t('entfernung_bahn'),
            min_value=0.0,
            max_value=10.0,
            value=0.5,
            step=0.1
        )
        
        col1, col2 = st.columns(2)
        with col1:
            garten = st.checkbox(t('garten'), value=True)
        with col2:
            balkon = st.checkbox(t('balkon'), value=True)
        
        # Buttons
        col1, col2 = st.columns(2)
        with col1:
            vorhersagen = st.form_submit_button(t('vorhersagen'), use_container_width=True)
        with col2:
            zurücksetzen = st.form_submit_button(t('zurücksetzen'), use_container_width=True)
    
    # Beispiel laden (außerhalb des Forms)
    if st.button(t('beispiel_laden'), use_container_width=True):
        st.session_state['beispiel_geladen'] = True
        st.rerun()

# ============================================
# HAUPTBEREICH
# ============================================

# Titel
st.markdown(f"# {t('titel')}")
st.markdown(f"### {t('untertitel')}")
st.markdown("---")

# Wenn API offline, zeige Warnung
if not api_online:
    st.warning("⚠️ Die API ist nicht erreichbar. Bitte starte sie mit: `python api.py`")
    st.stop()

# ============================================
# VORHERSAGE DURCHFÜHREN
# ============================================

def mache_vorhersage():
    """Sendet die Daten an die API und erhält die Vorhersage"""
    
    # Daten vorbereiten
    daten = {
        "flaeche": flaeche,
        "zimmer": zimmer,
        "schlafzimmer": schlafzimmer,
        "etage": etage,
        "stadt": stadt,
        "plz": plz,
        "baujahr": baujahr,
        "entfernung_bahn": entfernung_bahn,
        "garten": 1 if garten else 0,
        "balkon": 1 if balkon else 0
    }
    
    try:
        response = requests.post(
            "http://localhost:8000/predict",
            json=daten,
            timeout=5
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Fehler {response.status_code}: {response.text}")
            return None
    except Exception as e:
        st.error(f"Verbindungsfehler: {e}")
        return None

# Vorhersage ausführen
if vorhersagen or st.session_state.get('beispiel_geladen', False):
    with st.spinner("🔮 Berechne Vorhersage..."):
        ergebnis = mache_vorhersage()
        
        if ergebnis:
            # Beispiel-Status zurücksetzen
            if st.session_state.get('beispiel_geladen', False):
                st.session_state['beispiel_geladen'] = False
            
            # ============================================
            # ERGEBNISSE ANZEIGEN
            # ============================================
            st.markdown(f"## {t('ergebnisse')}")
            
            # Hauptkarte mit Preis
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                            border-radius: 15px; padding: 25px; text-align: center;">
                    <h3 style="color: white; margin: 0;">{t('geschätzter_preis')}</h3>
                    <p style="color: white; font-size: 2.5rem; font-weight: bold; margin: 10px 0;">
                        {ergebnis['preis_formatiert']}
                    </p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div style="background: #f8f9fa; border-radius: 15px; padding: 25px; text-align: center;">
                    <h3 style="color: #667eea; margin: 0;">{t('preis_pro_m2')}</h3>
                    <p style="color: #333; font-size: 2rem; font-weight: bold; margin: 10px 0;">
                        {ergebnis['preis_pro_m2']:,.0f} €
                    </p>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                <div style="background: #f8f9fa; border-radius: 15px; padding: 25px; text-align: center;">
                    <h3 style="color: #667eea; margin: 0;">{t('intervall')}</h3>
                    <p style="color: #333; font-size: 1.2rem; margin: 10px 0;">
                        {ergebnis['interval_min']:,.0f} € - {ergebnis['interval_max']:,.0f} €
                    </p>
                    <small style="color: #999;">90% Konfidenzintervall</small>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("---")
            
            # ============================================
            # VISUALISIERUNGEN
            # ============================================
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"### {t('feature_wichtigkeit')}")
                
                # Feature Importance Diagramm
                importance = get_feature_importance()
                fig = px.bar(
                    x=list(importance.values()),
                    y=list(importance.keys()),
                    orientation='h',
                    title=t('feature_wichtigkeit'),
                    labels={'x': 'Wichtigkeit', 'y': ''},
                    color=list(importance.values()),
                    color_continuous_scale='Purples'
                )
                fig.update_layout(showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.markdown(f"### {t('informationen')}")
                
                with st.expander(f"📊 {t('über_das_modell')}"):
                    st.markdown(f"""
                    {t('modell_beschreibung')}
                    
                    **{t('feature_beschreibung')}**
                    
                    **{t('genauigkeit')}**
                    """)
                
                # Vergleich mit ähnlichen Immobilien
                st.markdown("### 📈 Preisvergleich")
                
                # Beispiel-Daten für Vergleich
                preise_vergleich = {
                    'Ihre Immobilie': ergebnis['preis_estimate'],
                    'Durchschnitt (Stadt)': ergebnis['preis_estimate'] * 0.95,
                    'Durchschnitt (Region)': ergebnis['preis_estimate'] * 0.85
                }
                
                fig = px.bar(
                    x=list(preise_vergleich.keys()),
                    y=list(preise_vergleich.values()),
                    title="Preisvergleich",
                    labels={'x': '', 'y': 'Preis (€)'},
                    color=list(preise_vergleich.values()),
                    color_continuous_scale='Viridis'
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # ============================================
            # EINGABEDATEN ZUSAMMENFASSUNG
            # ============================================
            
            with st.expander("📋 Eingabedaten Zusammenfassung"):
                df_eingabe = pd.DataFrame([ergebnis['features']])
                st.dataframe(df_eingabe.T.rename(columns={0: 'Wert'}), use_container_width=True)

# ============================================
# FUSSZEILE
# ============================================

st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #999; padding: 20px;">
    <small>
        © 2024 Immobilienpreis-Vorhersage | 
        <a href="https://github.com/dein-repo" style="color: #667eea;">GitHub</a> |
        Version 1.0.0
    </small>
</div>
""", unsafe_allow_html=True)