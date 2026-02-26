# api.py
# ============================================
# FASTAPI BACKEND FÜR IMMOBILIENPREISVORHERSAGE
# ============================================

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import joblib
import pandas as pd
import numpy as np
import os
from typing import Optional, List, Dict
import uvicorn

# ============================================
# API INITIALISIERUNG
# ============================================

app = FastAPI(
    title="Immobilienpreis-Vorhersage API",
    description="API zur Vorhersage von Immobilienpreisen in Deutschland",
    version="1.0.0"
)

# CORS für Streamlit-Frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================
# MODELL LADEN
# ============================================

MODEL_PFAD = "modelle/random_forest_model.pkl"
SCALER_PFAD = "modelle/scaler.pkl"
FEATURES_PFAD = "modelle/feature_cols.pkl"

model = None
scaler = None
feature_cols = None

@app.on_event("startup")
async def lade_modelle():
    """Lädt die Modelle beim Start der API"""
    global model, scaler, feature_cols
    
    try:
        if os.path.exists(MODEL_PFAD):
            model = joblib.load(MODEL_PFAD)
            print(f"✅ Modell geladen: {MODEL_PFAD}")
        else:
            print(f"⚠️ Modell nicht gefunden: {MODEL_PFAD}")
        
        if os.path.exists(SCALER_PFAD):
            scaler = joblib.load(SCALER_PFAD)
            print(f"✅ Scaler geladen: {SCALER_PFAD}")
        
        if os.path.exists(FEATURES_PFAD):
            feature_cols = joblib.load(FEATURES_PFAD)
            print(f"✅ Features geladen: {len(feature_cols)} Features")
    except Exception as e:
        print(f"❌ Fehler beim Laden: {e}")

# ============================================
# PYDANTIC MODELLE
# ============================================

class ImmobilieInput(BaseModel):
    """Eingabedaten für eine Immobilie"""
    flaeche: float = Field(..., gt=0, description="Wohnfläche in m²", example=85)
    zimmer: int = Field(..., gt=0, description="Anzahl Zimmer", example=4)
    schlafzimmer: int = Field(..., gt=0, description="Anzahl Schlafzimmer", example=2)
    etage: int = Field(..., ge=0, description="Etage", example=2)
    stadt: str = Field(..., description="Stadt", example="Berlin")
    plz: int = Field(..., ge=10000, le=99999, description="Postleitzahl", example=10115)
    baujahr: int = Field(..., ge=1900, le=2024, description="Baujahr", example=2010)
    entfernung_bahn: float = Field(..., ge=0, le=10, description="Entfernung zum Bahnhof (km)", example=0.5)
    garten: int = Field(..., ge=0, le=1, description="Garten (0=nein, 1=ja)", example=1)
    balkon: int = Field(..., ge=0, le=1, description="Balkon (0=nein, 1=ja)", example=1)
    
    class Config:
        json_schema_extra = {
            "example": {
                "flaeche": 85,
                "zimmer": 4,
                "schlafzimmer": 2,
                "etage": 2,
                "stadt": "Berlin",
                "plz": 10115,
                "baujahr": 2010,
                "entfernung_bahn": 0.5,
                "garten": 1,
                "balkon": 1
            }
        }

class VorhersageResponse(BaseModel):
    """Antwort mit der Vorhersage"""
    preis_estimate: float = Field(..., description="Geschätzter Preis in €")
    preis_formatiert: str = Field(..., description="Formatierter Preis")
    preis_pro_m2: float = Field(..., description="Geschätzter Preis pro m²")
    interval_min: float = Field(..., description="Unteres Intervall")
    interval_max: float = Field(..., description="Oberes Intervall")
    modell: str = Field(..., description="Verwendetes Modell")
    features: Dict = Field(..., description="Verwendete Features")

class GesundheitsResponse(BaseModel):
    """Gesundheitsstatus der API"""
    status: str
    modell_geladen: bool
    scaler_geladen: bool
    features_geladen: bool
    anzahl_features: int
    version: str

# ============================================
# FEATURE ENGINEERING
# ============================================

def erstelle_features(df):
    """
    Erstellt Features für die Vorhersage (gleiche Logik wie im Training)
    """
    df = df.copy()
    
    # Alter berechnen
    aktuelles_jahr = 2024
    df['alter'] = aktuelles_jahr - df['baujahr']
    
    # Zimmer pro Fläche
    df['zimmer_pro_m2'] = df['zimmer'] / df['flaeche'] * 100
    
    # Komfort-Score
    df['komfort_score'] = df['garten'] + df['balkon'] + (df['etage'] > 0).astype(int)
    
    # One-Hot-Encoding für Städte
    staedte = ['Berlin', 'Hamburg', 'München', 'Köln', 'Frankfurt', 'Stuttgart']
    for stadt in staedte:
        df[f'stadt_{stadt}'] = (df['stadt'] == stadt).astype(int)
    
    return df

# ============================================
# API ENDPUNKTE
# ============================================

@app.get("/", response_model=GesundheitsResponse)
async def root():
    """Root-Endpunkt - Gesundheitscheck"""
    return GesundheitsResponse(
        status="online",
        modell_geladen=model is not None,
        scaler_geladen=scaler is not None,
        features_geladen=feature_cols is not None,
        anzahl_features=len(feature_cols) if feature_cols else 0,
        version="1.0.0"
    )

@app.get("/health")
async def health():
    """Einfacher Gesundheitscheck"""
    if model is None or scaler is None:
        raise HTTPException(status_code=503, detail="Modell nicht geladen")
    return {"status": "ok", "message": "API ist bereit"}

@app.post("/predict", response_model=VorhersageResponse)
async def predict(immobilie: ImmobilieInput):
    """
    Sagt den Preis einer Immobilie vorher
    """
    global model, scaler, feature_cols
    
    # Prüfen ob Modell geladen
    if model is None or scaler is None or feature_cols is None:
        raise HTTPException(status_code=503, detail="Modell nicht verfügbar")
    
    try:
        # Eingabe in DataFrame konvertieren
        df_input = pd.DataFrame([immobilie.dict()])
        
        # Feature Engineering
        df_features = erstelle_features(df_input)
        
        # Sicherstellen, dass alle benötigten Spalten vorhanden sind
        for col in feature_cols:
            if col not in df_features.columns:
                df_features[col] = 0
        
        # Richtige Reihenfolge der Features
        df_features = df_features[feature_cols]
        
        # Skalieren
        X_scaled = scaler.transform(df_features)
        
        # Vorhersage
        preis = model.predict(X_scaled)[0]
        
        # Intervall (vereinfacht)
        interval = preis * 0.1  # 10% Unsicherheit
        
        # Formatierung
        preis_formatiert = f"{preis:,.0f} €".replace(",", ".")
        
        return VorhersageResponse(
            preis_estimate=round(preis, 2),
            preis_formatiert=preis_formatiert,
            preis_pro_m2=round(preis / immobilie.flaeche, 2),
            interval_min=round(preis - interval, 2),
            interval_max=round(preis + interval, 2),
            modell="Random Forest",
            features=immobilie.dict()
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Vorhersage fehlgeschlagen: {str(e)}")

@app.get("/features")
async def get_features():
    """Listet alle verwendeten Features auf"""
    if feature_cols:
        return {"features": feature_cols, "anzahl": len(feature_cols)}
    return {"features": [], "anzahl": 0}

# ============================================
# SERVER START
# ============================================

if __name__ == "__main__":
    print("="*60)
    print("🏠 IMMOBILIENPREIS API STARTEN")
    print("="*60)
    print("\n📡 Server läuft auf: http://localhost:8000")
    print("📚 Dokumentation: http://localhost:8000/docs")
    print("\nDrücke STRG+C zum Beenden")
    print("="*60)
    
    uvicorn.run(app, host="0.0.0.0", port=8000)