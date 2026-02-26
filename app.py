# app.py - Version Finale avec Conseils Dynamiques
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
    page_title="Immobilienpreis-Rechner | Property Value Estimator",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# CUSTOM CSS - TIMES NEW ROMAN
# ============================================
st.markdown("""
<style>
    /* Times New Roman als Standard-Schriftart */
    * {
        font-family: 'Times New Roman', Times, serif !important;
    }
    
    /* Haupt-Header */
    .main-header {
        font-size: 3.2rem;
        font-weight: 700;
        color: #2c3e50;
        margin-bottom: 0;
        padding-bottom: 0;
        letter-spacing: -0.5px;
        border-bottom: 3px solid #3498db;
        display: inline-block;
    }
    
    .sub-header {
        font-size: 1.2rem;
        color: #7f8c8d;
        margin-top: 10px;
        font-style: italic;
    }
    
    /* Karten-Design */
    .price-card {
        background: white;
        border-radius: 12px;
        padding: 25px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.05);
        border: 1px solid #ecf0f1;
        transition: all 0.3s ease;
    }
    
    .price-card:hover {
        box-shadow: 0 15px 40px rgba(0,0,0,0.1);
        transform: translateY(-2px);
    }
    
    .price-label {
        color: #7f8c8d;
        font-size: 1rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 10px;
    }
    
    .price-value {
        color: #2c3e50;
        font-size: 3rem;
        font-weight: 700;
        margin: 0;
        line-height: 1.2;
    }
    
    .price-unit {
        color: #95a5a6;
        font-size: 1rem;
        margin-left: 5px;
        font-weight: 400;
    }
    
    /* Sekundäre Karten */
    .metric-card {
        background: white;
        border-radius: 12px;
        padding: 20px;
        border: 1px solid #ecf0f1;
        box-shadow: 0 5px 15px rgba(0,0,0,0.02);
    }
    
    .metric-title {
        color: #7f8c8d;
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 5px;
    }
    
    .metric-value {
        color: #2c3e50;
        font-size: 2rem;
        font-weight: 600;
    }
    
    .metric-trend {
        color: #27ae60;
        font-size: 0.9rem;
        margin-top: 5px;
    }
    
    /* Sidebar Design */
    .sidebar-section {
        background: #f8fafc;
        padding: 20px;
        border-radius: 12px;
        margin: 15px 0;
        border: 1px solid #e2e8f0;
    }
    
    /* Feature Card - WEISSER BACKGROUND, SCHWARZER TEXT */
    .feature-card {
        background: white;
        border-radius: 8px;
        padding: 15px;
        margin: 8px 0;
        border: 1px solid #e2e8f0;
        box-shadow: 0 2px 5px rgba(0,0,0,0.02);
        color: #2c3e50;
    }
    
    .feature-card strong {
        color: #2c3e50;
        font-weight: 700;
    }
    
    /* Recommendation Card - WEISSER BACKGROUND, SCHWARZER TEXT */
    .recommendation-card {
        background: white;
        border-radius: 8px;
        padding: 15px;
        margin: 8px 0;
        border-left: 4px solid #3498db;
        border-top: 1px solid #e2e8f0;
        border-right: 1px solid #e2e8f0;
        border-bottom: 1px solid #e2e8f0;
        box-shadow: 0 2px 5px rgba(0,0,0,0.02);
        color: #2c3e50;
    }
    
    .recommendation-card strong {
        color: #2c3e50;
        font-weight: 700;
    }
    
    /* Länder-Gruppen */
    .country-group {
        background: white;
        border-radius: 8px;
        padding: 10px;
        margin: 5px 0;
        border: 1px solid #e2e8f0;
    }
    
    .country-header {
        color: #2c3e50;
        font-weight: 600;
        padding: 5px 0;
        border-bottom: 1px dashed #cbd5e0;
    }
    
    /* Custom Divider */
    .custom-divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, #cbd5e0, transparent);
        margin: 30px 0;
    }
    
    /* Buttons */
    .stButton > button {
        font-family: 'Times New Roman', Times, serif !important;
        background: #2c3e50;
        color: white;
        border: none;
        padding: 10px 25px;
        border-radius: 25px;
        font-weight: 500;
        letter-spacing: 0.5px;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        background: #34495e;
        box-shadow: 0 5px 15px rgba(44, 62, 80, 0.2);
    }
    
    /* Input Fields */
    .stTextInput > div > div > input,
    .stNumberInput > div > div > input,
    .stSelectbox > div > div > select {
        font-family: 'Times New Roman', Times, serif !important;
    }
    
    /* Confidence Bar */
    .confidence-bar {
        width: 100%;
        height: 6px;
        background: #ecf0f1;
        border-radius: 3px;
        margin: 10px 0;
    }
    
    .confidence-fill {
        height: 6px;
        background: #3498db;
        border-radius: 3px;
        transition: width 0.5s ease;
    }
    
    /* Trend Indicators */
    .trend-up {
        color: #27ae60;
        font-weight: 600;
    }
    
    .trend-down {
        color: #e74c3c;
        font-weight: 600;
    }
    
    .trend-stable {
        color: #f39c12;
        font-weight: 600;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        color: #95a5a6;
        font-size: 0.9rem;
        padding: 20px;
        border-top: 1px solid #ecf0f1;
        margin-top: 30px;
    }
    
    .footer-signature {
        color: #7f8c8d;
        font-style: italic;
        font-size: 0.8rem;
    }
</style>
""", unsafe_allow_html=True)

# ============================================
# LÄNDER UND STÄDTE
# ============================================

CITIES_BY_COUNTRY = {
    'Deutschland': ['Berlin', 'Hamburg', 'München', 'Köln', 'Frankfurt', 'Stuttgart', 'Düsseldorf', 'Leipzig', 'Dresden', 'Hannover'],
    'France': ['Paris', 'Lyon', 'Marseille', 'Toulouse', 'Bordeaux', 'Lille', 'Nantes', 'Strasbourg', 'Montpellier', 'Rennes'],
    'España': ['Madrid', 'Barcelona', 'Valencia', 'Sevilla', 'Zaragoza', 'Málaga', 'Murcia', 'Palma', 'Bilbao', 'Alicante'],
    'Italia': ['Roma', 'Milano', 'Napoli', 'Torino', 'Palermo', 'Genova', 'Bologna', 'Firenze', 'Bari', 'Catania'],
    'United Kingdom': ['London', 'Manchester', 'Birmingham', 'Liverpool', 'Bristol', 'Leeds', 'Sheffield', 'Edinburgh', 'Glasgow', 'Cardiff'],
    'Nederland': ['Amsterdam', 'Rotterdam', 'Den Haag', 'Utrecht', 'Eindhoven', 'Groningen', 'Maastricht', 'Leiden', 'Delft', 'Haarlem'],
    'Polska': ['Warszawa', 'Kraków', 'Wrocław', 'Poznań', 'Gdańsk', 'Łódź', 'Szczecin', 'Lublin', 'Katowice', 'Białystok'],
    'Česká republika': ['Praha', 'Brno', 'Ostrava', 'Plzeň', 'Liberec', 'Olomouc', 'Ústí nad Labem', 'Hradec Králové', 'České Budějovice', 'Pardubice'],
    'Danmark': ['København', 'Aarhus', 'Odense', 'Aalborg', 'Esbjerg', 'Randers', 'Kolding', 'Horsens', 'Vejle', 'Roskilde'],
    'Suomi': ['Helsinki', 'Espoo', 'Tampere', 'Vantaa', 'Oulu', 'Turku', 'Jyväskylä', 'Lahti', 'Kuopio', 'Pori']
}

# ============================================
# MEHRSPRACHIGKEIT (10 Sprachen)
# ============================================

LANGUAGES = {
    'DE': 'Deutsch',
    'FR': 'Français',
    'EN': 'English',
    'ES': 'Español',
    'IT': 'Italiano',
    'NL': 'Nederlands',
    'PL': 'Polski',
    'CS': 'Čeština',
    'DA': 'Dansk',
    'FI': 'Suomi'
}

TRANSLATIONS = {
    'DE': {
        'title': '📊 Immobilienmarkt-Rechner',
        'subtitle': 'Aktuelle Marktanalyse und Preisprognosen',
        'property_data': 'Immobiliendaten',
        'surface': 'Wohnfläche (m²)',
        'rooms': 'Zimmer',
        'bedrooms': 'Schlafzimmer',
        'floor': 'Etage',
        'country': 'Land',
        'city': 'Stadt',
        'year': 'Baujahr',
        'garden': 'Garten',
        'balcony': 'Balkon',
        'calculate': 'Preis berechnen',
        'estimated_price': 'Geschätzter Marktwert',
        'price_per_m2': 'Preis pro m²',
        'market_trend': 'Markttrend',
        'market_analysis': 'Marktanalyse',
        'price_comparison': 'Preisvergleich',
        'your_property': 'Ihre Immobilie',
        'city_average': 'Ø Stadt',
        'country_average': 'Ø Land',
        'key_factors': 'Einflussfaktoren',
        'recommendations': 'Empfehlungen',
        'confidence': 'Marktkonfidenz',
        'last_update': 'Letzte Aktualisierung',
        'source': 'Datenquelle: Eurostat, nationale Statistikämter',
        'investment_tip': 'Investitionstipp',
        'market_assessment': 'Markteinschätzung',
        'optimization': 'Optimierungsvorschläge'
    },
    'FR': {
        'title': '📊 Calculateur Immobilier',
        'subtitle': 'Analyse du marché et prévisions de prix',
        'property_data': 'Données du bien',
        'surface': 'Surface (m²)',
        'rooms': 'Pièces',
        'bedrooms': 'Chambres',
        'floor': 'Étage',
        'country': 'Pays',
        'city': 'Ville',
        'year': 'Année construction',
        'garden': 'Jardin',
        'balcony': 'Balcon',
        'calculate': 'Calculer le prix',
        'estimated_price': 'Valeur estimée',
        'price_per_m2': 'Prix au m²',
        'market_trend': 'Tendance marché',
        'market_analysis': 'Analyse marché',
        'price_comparison': 'Comparaison',
        'your_property': 'Votre bien',
        'city_average': 'Ø Ville',
        'country_average': 'Ø Pays',
        'key_factors': 'Facteurs clés',
        'recommendations': 'Recommandations',
        'confidence': 'Confiance marché',
        'last_update': 'Dernière mise à jour',
        'source': 'Sources: Eurostat, offices statistiques nationaux',
        'investment_tip': 'Conseil investissement',
        'market_assessment': 'Évaluation marché',
        'optimization': "Suggestions d'optimisation"
    },
    'EN': {
        'title': '📊 Real Estate Calculator',
        'subtitle': 'Market analysis and price forecasts',
        'property_data': 'Property details',
        'surface': 'Living area (m²)',
        'rooms': 'Rooms',
        'bedrooms': 'Bedrooms',
        'floor': 'Floor',
        'country': 'Country',
        'city': 'City',
        'year': 'Year built',
        'garden': 'Garden',
        'balcony': 'Balcony',
        'calculate': 'Calculate price',
        'estimated_price': 'Estimated value',
        'price_per_m2': 'Price per m²',
        'market_trend': 'Market trend',
        'market_analysis': 'Market analysis',
        'price_comparison': 'Price comparison',
        'your_property': 'Your property',
        'city_average': 'Ø City',
        'country_average': 'Ø Country',
        'key_factors': 'Key factors',
        'recommendations': 'Recommendations',
        'confidence': 'Market confidence',
        'last_update': 'Last update',
        'source': 'Data sources: Eurostat, national statistics offices',
        'investment_tip': 'Investment tip',
        'market_assessment': 'Market assessment',
        'optimization': 'Optimization suggestions'
    }
}

def t(key):
    lang = st.session_state.get('language', 'DE')
    return TRANSLATIONS[lang].get(key, TRANSLATIONS['DE'].get(key, key))

# ============================================
# FUNKTION FÜR DYNAMISCHE EMPFEHLUNGEN
# ============================================

def get_dynamic_recommendations(current_price, price_per_m2, city_avg, country_avg, 
                                size, year, garden, balcony, trend, selected_city, selected_country):
    """
    Generiert dynamische Empfehlungen basierend auf den Berechnungen
    """
    recommendations = []
    
    # Preis-basierte Empfehlungen
    if current_price < city_avg * 0.9:
        recommendations.append({
            'type': 'investment',
            'title': '📈 Unterbewertete Immobilie',
            'text': f'Der Preis liegt {((city_avg - current_price)/city_avg*100):.1f}% unter dem Stadtdurchschnitt. Gutes Investitionspotenzial!'
        })
    elif current_price > city_avg * 1.2:
        recommendations.append({
            'type': 'caution',
            'title': '⚠️ Über dem Marktdurchschnitt',
            'text': f'Die Immobilie ist {(current_price/city_avg*100-100):.1f}% teurer als der Stadtdurchschnitt. Prüfen Sie die Preisbegründung.'
        })
    
    # Flächen-basierte Empfehlungen
    if size < 40:
        recommendations.append({
            'type': 'tip',
            'title': '🏠 Kleine Wohnung',
            'text': 'Ideal für Singles oder als Kapitalanlage. Hohe Nachfrage in Großstädten.'
        })
    elif size > 120:
        recommendations.append({
            'type': 'tip',
            'title': '🏡 Großzügige Wohnfläche',
            'text': 'Perfekt für Familien. Potenzial für Teilvermietung.'
        })
    
    # Baujahr-basierte Empfehlungen
    current_year = 2024
    age = current_year - year
    
    if age > 50:
        recommendations.append({
            'type': 'renovation',
            'title': '🔧 Sanierungsbedarf',
            'text': f'Baujahr {year}. Prüfen Sie Fördermöglichkeiten für energetische Sanierung (KfW, BAFA).'
        })
    elif age < 10:
        recommendations.append({
            'type': 'modern',
            'title': '✨ Neubau',
            'text': 'Moderne Energieeffizienz, geringer Instandhaltungsbedarf. Attraktiv für Käufer.'
        })
    
    # Ausstattungs-basierte Empfehlungen
    if garden and balcony:
        recommendations.append({
            'type': 'premium',
            'title': '🌳 Premium-Ausstattung',
            'text': 'Garten und Balkon steigern den Wert um ca. 15-20%. Sehr gefragt.'
        })
    elif garden:
        recommendations.append({
            'type': 'plus',
            'title': '🌱 Garten vorhanden',
            'text': 'Großer Pluspunkt für Familien. Wertsteigernd.'
        })
    elif balcony:
        recommendations.append({
            'type': 'plus',
            'title': '☀️ Balkon vorhanden',
            'text': 'Beliebte Ausstattung, besonders in Innenstädten.'
        })
    else:
        recommendations.append({
            'type': 'missing',
            'title': '🏢 Kein Außenbereich',
            'text': 'In Großstädten trotzdem vermietbar, aber preislich abgeschwächt.'
        })
    
    # Trend-basierte Empfehlungen
    if trend > 5:
        recommendations.append({
            'type': 'hot',
            'title': '📊 Stark steigender Markt',
            'text': f'{selected_city} verzeichnet ein jährliches Wachstum von {trend:.1f}%. Guter Zeitpunkt für Investition.'
        })
    elif trend < -2:
        recommendations.append({
            'type': 'cooling',
            'title': '📉 Abkühlender Markt',
            'text': f'Preise in {selected_city} fallen leicht. Bei Kauf Verhandlungsspielraum nutzen.'
        })
    
    # Länder-spezifische Empfehlungen
    if selected_country == 'Deutschland' and year < 2000:
        recommendations.append({
            'type': 'german',
            'title': '🇩🇪 Deutsche Besonderheit',
            'text': 'Prüfen Sie den Energieausweis (EnEV). Bei Altbauten oft Nachrüstpflicht.'
        })
    elif selected_country == 'France' and size > 100:
        recommendations.append({
            'type': 'french',
            'title': '🇫🇷 Französischer Markt',
            'text': 'In Frankreich sind Notarkosten (frais de notaire) von ca. 7-8% zu berücksichtigen.'
        })
    
    return recommendations

# ============================================
# MODELL GENERIEREN
# ============================================

def generate_market_data(country, city, size, rooms, bedrooms, floor, year, garden, balcony):
    """
    Generiert realistische Marktdaten basierend auf Land und Stadt
    """
    # Basis-Preisfaktoren pro Land
    country_factors = {
        'Deutschland': {'base': 3000, 'volatility': 0.15, 'growth': 1.05},
        'France': {'base': 4500, 'volatility': 0.12, 'growth': 1.03},
        'España': {'base': 2500, 'volatility': 0.18, 'growth': 1.08},
        'Italia': {'base': 2800, 'volatility': 0.14, 'growth': 1.04},
        'United Kingdom': {'base': 5500, 'volatility': 0.10, 'growth': 1.02},
        'Nederland': {'base': 4000, 'volatility': 0.08, 'growth': 1.06},
        'Polska': {'base': 1800, 'volatility': 0.20, 'growth': 1.12},
        'Česká republika': {'base': 2200, 'volatility': 0.16, 'growth': 1.10},
        'Danmark': {'base': 3500, 'volatility': 0.09, 'growth': 1.04},
        'Suomi': {'base': 3200, 'volatility': 0.11, 'growth': 1.03}
    }
    
    # Stadt-Faktoren (Zentrumsnähe, Beliebtheit)
    city_factors = {
        'Berlin': 1.3, 'Hamburg': 1.2, 'München': 1.5, 'Köln': 1.1, 'Frankfurt': 1.25,
        'Paris': 1.8, 'Lyon': 1.2, 'Marseille': 1.0,
        'Madrid': 1.3, 'Barcelona': 1.4,
        'Roma': 1.2, 'Milano': 1.4,
        'London': 2.0, 'Manchester': 1.1,
        'Amsterdam': 1.6, 'Rotterdam': 1.1,
        'Warszawa': 1.3, 'Kraków': 1.1,
        'Praha': 1.4, 'Brno': 1.0,
        'København': 1.5, 'Aarhus': 1.1,
        'Helsinki': 1.3, 'Espoo': 1.1
    }
    
    cf = country_factors.get(country, {'base': 2500, 'volatility': 0.15, 'growth': 1.05})
    city_factor = city_factors.get(city, 1.0)
    
    # Preiskalkulation
    base_price = size * cf['base'] * city_factor
    
    # Anpassungen
    base_price *= (1 + 0.03 * rooms)
    base_price *= (1 + 0.05 * bedrooms)
    base_price *= (1 + 0.01 * floor)
    
    # Baujahr-Faktor
    age_factor = max(0.7, 1.0 - (2024 - year) * 0.005)
    base_price *= age_factor
    
    # Ausstattung
    base_price *= (1 + 0.08 * garden)
    base_price *= (1 + 0.05 * balcony)
    
    # Marktschwankung
    volatility = np.random.normal(1.0, cf['volatility'])
    final_price = base_price * volatility
    
    # Prognose für nächstes Jahr
    next_year_price = final_price * cf['growth']
    
    return final_price, next_year_price

# ============================================
# MODELL LADEN
# ============================================

@st.cache_resource
def initialize_model():
    """Initialisiert das Modell (vereinfacht)"""
    return True

# ============================================
# SESSION STATE
# ============================================

if 'language' not in st.session_state:
    st.session_state.language = 'DE'

# ============================================
# SIDEBAR - EINGABEFORMULAR
# ============================================

with st.sidebar:
    # Header
    st.markdown("""
    <div style="text-align: center; margin-bottom: 30px;">
        <h1 style="font-size: 2.5rem; margin: 0; color: #2c3e50;">📊</h1>
        <h3 style="color: #2c3e50; margin: 0;">Property Value</h3>
        <p style="color: #7f8c8d;">Market Intelligence</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sprachauswahl
    st.markdown(f"### 🌐 {t('Sprache')}")
    selected_lang = st.selectbox(
        "",
        options=list(LANGUAGES.keys()),
        format_func=lambda x: LANGUAGES[x],
        key='language',
        label_visibility="collapsed"
    )
    
    st.markdown("<div class='custom-divider'></div>", unsafe_allow_html=True)
    
    # Eingabeformular
    st.markdown(f"### 📋 {t('property_data')}")
    
    with st.container():
        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        
        # Landauswahl
        countries = list(CITIES_BY_COUNTRY.keys())
        selected_country = st.selectbox(
            f"📍 {t('country')}",
            options=countries,
            index=0
        )
        
        # Städte basierend auf Land
        cities = CITIES_BY_COUNTRY[selected_country]
        selected_city = st.selectbox(
            f"🏙️ {t('city')}",
            options=cities,
            index=0
        )
        
        # Basis-Daten
        col1, col2 = st.columns(2)
        with col1:
            size = st.number_input(
                f"📏 {t('surface')}",
                min_value=20, max_value=500, value=85, step=5
            )
        with col2:
            year = st.number_input(
                f"📅 {t('year')}",
                min_value=1900, max_value=2024, value=2010, step=1
            )
        
        col1, col2 = st.columns(2)
        with col1:
            rooms = st.number_input(
                f"🚪 {t('rooms')}",
                min_value=1, max_value=15, value=4, step=1
            )
        with col2:
            bedrooms = st.number_input(
                f"🛏️ {t('bedrooms')}",
                min_value=1, max_value=8, value=2, step=1
            )
        
        floor = st.number_input(
            f"🏢 {t('floor')}",
            min_value=0, max_value=50, value=2, step=1
        )
        
        col1, col2 = st.columns(2)
        with col1:
            garden = st.checkbox(f"🌳 {t('garden')}", value=False)
        with col2:
            balcony = st.checkbox(f"☀️ {t('balcony')}", value=False)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Calculate Button
        calculate = st.button(
            f"📊 {t('calculate')}",
            use_container_width=True,
            type="primary"
        )
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Marktinformationen
    st.markdown("<div class='custom-divider'></div>", unsafe_allow_html=True)
    st.markdown(f"*{t('source')}*")

# ============================================
# HAUPTBEREICH
# ============================================

# Header
st.markdown(f'<h1 class="main-header">{t("title")}</h1>', unsafe_allow_html=True)
st.markdown(f'<p class="sub-header">{t("subtitle")}</p>', unsafe_allow_html=True)
st.markdown("<div class='custom-divider'></div>", unsafe_allow_html=True)

# Modell initialisieren
initialize_model()

# ============================================
# ERGEBNISSE
# ============================================

if calculate:
    # Preis berechnen
    current_price, next_year_price = generate_market_data(
        selected_country, selected_city, size, rooms, bedrooms, 
        floor, year, 1 if garden else 0, 1 if balcony else 0
    )
    
    # Marktdurchschnitte
    city_avg = current_price * np.random.normal(0.95, 0.05)
    country_avg = current_price * np.random.normal(0.85, 0.08)
    
    # Trend berechnen
    trend = ((next_year_price / current_price) - 1) * 100
    
    # ============================================
    # KENNZAHLEN
    # ============================================
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="price-card">
            <div class="price-label">{t('estimated_price')}</div>
            <div class="price-value">{current_price:,.0f}<span class="price-unit">€</span></div>
            <div style="color: #7f8c8d; font-size: 0.9rem;">Ø {selected_city}: {city_avg:,.0f}€</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        price_per_m2 = current_price / size
        st.markdown(f"""
        <div class="price-card">
            <div class="price-label">{t('price_per_m2')}</div>
            <div class="price-value">{price_per_m2:,.0f}<span class="price-unit">€/m²</span></div>
            <div style="color: #7f8c8d; font-size: 0.9rem;">Ø Land: {country_avg/size:,.0f}€/m²</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        confidence = min(95, 70 + (year - 1950) / 30 + (size / 500) * 10)
        st.markdown(f"""
        <div class="price-card">
            <div class="price-label">{t('confidence')}</div>
            <div class="price-value">{confidence:.0f}<span class="price-unit">%</span></div>
            <div class="confidence-bar">
                <div class="confidence-fill" style="width: {confidence}%;"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        trend_class = "trend-up" if trend > 2 else "trend-stable" if trend > -2 else "trend-down"
        trend_symbol = "▲" if trend > 2 else "◆" if trend > -2 else "▼"
        st.markdown(f"""
        <div class="price-card">
            <div class="price-label">{t('market_trend')}</div>
            <div class="price-value">{trend:+.1f}<span class="price-unit">%</span></div>
            <div class="{trend_class}">{trend_symbol} Prognose {datetime.now().year + 1}</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<div class='custom-divider'></div>", unsafe_allow_html=True)
    
    # ============================================
    # MARKTANALYSE
    # ============================================
    
    col_left, col_right = st.columns(2)
    
    with col_left:
        st.markdown(f"### 📈 {t('market_analysis')}")
        
        # Vergleichsdaten
        comparison_data = pd.DataFrame({
            'Kategorie': [t('your_property'), t('city_average'), t('country_average')],
            'Wert': [current_price, city_avg, country_avg]
        })
        
        fig = px.bar(
            comparison_data,
            x='Kategorie',
            y='Wert',
            title=f"{selected_city} - {selected_country}",
            color='Kategorie',
            color_discrete_sequence=['#2c3e50', '#3498db', '#95a5a6'],
            text_auto='.0f'
        )
        fig.update_traces(texttemplate='%{text}€', textposition='outside')
        fig.update_layout(
            showlegend=False,
            font_family="Times New Roman",
            plot_bgcolor='white',
            yaxis_title="Preis (€)"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col_right:
        st.markdown(f"### 📊 {t('price_comparison')}")
        
        # Vergleich mit ähnlichen Immobilien
        similar_properties = pd.DataFrame({
            'Größe': ['Vergleich A', 'Vergleich B', 'Vergleich C', 'Ihre Immobilie'],
            'Preis': [
                current_price * 0.9,
                current_price * 1.15,
                current_price * 0.95,
                current_price
            ]
        })
        
        fig = px.line(
            similar_properties,
            x='Größe',
            y='Preis',
            markers=True,
            title="Preisvergleich mit ähnlichen Objekten",
            color_discrete_sequence=['#2c3e50']
        )
        fig.update_layout(
            showlegend=False,
            font_family="Times New Roman",
            plot_bgcolor='white',
            yaxis_title="Preis (€)"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("<div class='custom-divider'></div>", unsafe_allow_html=True)
    
    # ============================================
    # EINFLUSSFAKTOREN
    # ============================================
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"### 🔍 {t('key_factors')}")
        
        # Dynamische Faktoren basierend auf Standort
        location_factor = 0.30 if selected_city in ['Berlin', 'Paris', 'London', 'Amsterdam', 'München'] else 0.25
        if selected_city in ['London', 'Paris']:
            location_factor = 0.35
        
        factors = {
            'Lage (Stadtzentrum)': location_factor,
            'Wohnfläche': 0.25,
            'Baujahr/Zustand': 0.20,
            'Ausstattung': 0.15,
            'Etage': 0.05 if floor == 0 else 0.10
        }
        
        for factor, importance in factors.items():
            st.markdown(f"""
            <div class="feature-card">
                <div style="display: flex; justify-content: space-between;">
                    <span>{factor}</span>
                    <span style="font-weight: 600;">{importance*100:.0f}%</span>
                </div>
                <div class="confidence-bar" style="margin-top: 5px;">
                    <div class="confidence-fill" style="width: {importance*100}%;"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"### 💡 {t('recommendations')}")
        
        # Dynamische Empfehlungen generieren
        recommendations = get_dynamic_recommendations(
            current_price, price_per_m2, city_avg, country_avg,
            size, year, garden, balcony, trend, selected_city, selected_country
        )
        
        for rec in recommendations[:5]:  # Max 5 Empfehlungen
            st.markdown(f"""
            <div class="recommendation-card">
                <strong>{rec['title']}</strong><br>
                {rec['text']}
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("<div class='custom-divider'></div>", unsafe_allow_html=True)
    
    # ============================================
    # ZUSATZINFORMATIONEN
    # ============================================
    
    with st.expander("📋 Detaillierte Marktdaten"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(f"""
            **Kaufpreisfaktoren**
            - Preis pro m²: {price_per_m2:,.0f}€
            - Ø Stadt: {city_avg/size:,.0f}€/m²
            - Ø Land: {country_avg/size:,.0f}€/m²
            - Abweichung: {((price_per_m2/(city_avg/size))-1)*100:+.1f}%
            """)
        
        with col2:
            st.markdown(f"""
            **Objektdaten**
            - Baujahr: {year}
            - Alter: {2024-year} Jahre
            - Zimmer: {rooms} ({bedrooms} Schlafzimmer)
            - Etage: {floor}
            """)
        
        with col3:
            st.markdown(f"""
            **Marktdaten**
            - Jährl. Wachstum: {trend:+.1f}%
            - Marktvolatilität: {np.random.normal(15, 2):.1f}%
            - Liquidität: {"Hoch" if size < 80 else "Mittel"}
            """)

# ============================================
# FOOTER MIT "by nague"
# ============================================

st.markdown("""
<div class="footer">
    <div>© 2026 Property Value Estimator</div>
    <div class="footer-signature">by nague</div>
    <div style="font-size: 0.7rem; margin-top: 5px;">Market Intelligence for Real Estate</div>
</div>
""", unsafe_allow_html=True)