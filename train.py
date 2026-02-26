# train.py
# ============================================
# TRAINING-SKRIPT FÜR IMMOBILIENPREISVORHERSAGE
# ============================================

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import joblib
import os
import mlflow
import mlflow.sklearn
from datetime import datetime

# ============================================
# DATENGENERIERUNG
# ============================================

def generiere_daten(anzahl=1000):
    """
    Generiert synthetische Immobiliendaten für Deutschland
    """
    np.random.seed(42)
    
    # Features
    flaeche = np.random.normal(80, 30, anzahl).clip(30, 250).astype(int)
    zimmer = (flaeche / 25 + np.random.normal(0, 1, anzahl)).clip(1, 8).astype(int)
    schlafzimmer = (zimmer * 0.5 + np.random.normal(0, 1, anzahl)).clip(1, 5).astype(int)
    etage = np.random.randint(0, 10, anzahl)
    
    # Deutsche Postleitzahlen (vereinfacht)
    plz_regionen = {
        'Berlin': 10000, 'Hamburg': 20000, 'München': 80000,
        'Köln': 50000, 'Frankfurt': 60000, 'Stuttgart': 70000
    }
    stadt = np.random.choice(list(plz_regionen.keys()), anzahl)
    plz = [plz_regionen[s] + np.random.randint(0, 1000) for s in stadt]
    
    baujahr = np.random.randint(1950, 2023, anzahl)
    entfernung_bahn = np.random.uniform(0, 5, anzahl).round(1)
    garten = np.random.choice([0, 1], anzahl, p=[0.4, 0.6])
    balkon = np.random.choice([0, 1], anzahl, p=[0.3, 0.7])
    
    # Preisberechnung (realistisch für deutsche Städte)
    preis_pro_qm = 3000  # Basisp Preis
    preis_pro_qm += (stadt == 'München') * 4000
    preis_pro_qm += (stadt == 'Berlin') * 2000
    preis_pro_qm += (stadt == 'Hamburg') * 1500
    preis_pro_qm += (stadt == 'Frankfurt') * 1800
    
    preis = flaeche * preis_pro_qm
    preis *= (1 + 0.02 * etage)
    preis *= (1 + 0.1 * garten)
    preis *= (1 + 0.05 * balkon)
    preis *= (1 - 0.03 * entfernung_bahn)
    preis += np.random.normal(0, preis * 0.1, anzahl)
    preis = preis.clip(50000, 2000000).astype(int)
    
    # DataFrame
    df = pd.DataFrame({
        'flaeche': flaeche,
        'zimmer': zimmer,
        'schlafzimmer': schlafzimmer,
        'etage': etage,
        'stadt': stadt,
        'plz': plz,
        'baujahr': baujahr,
        'entfernung_bahn': entfernung_bahn,
        'garten': garten,
        'balkon': balkon,
        'preis': preis
    })
    
    return df

# ============================================
# FEATURE ENGINEERING
# ============================================

def erstelle_features(df):
    """
    Erstellt neue Features aus den Rohdaten
    """
    df = df.copy()
    
    # Preis pro m²
    df['preis_pro_m2'] = df['preis'] / df['flaeche']
    
    # Alter der Immobilie
    aktuelles_jahr = datetime.now().year
    df['alter'] = aktuelles_jahr - df['baujahr']
    
    # Zimmer pro Fläche
    df['zimmer_pro_m2'] = df['zimmer'] / df['flaeche'] * 100
    
    # Komfort-Score
    df['komfort_score'] = df['garten'] + df['balkon'] + (df['etage'] > 0).astype(int)
    
    # Stadt-Kategorien (für One-Hot-Encoding)
    df = pd.get_dummies(df, columns=['stadt'], prefix='stadt')
    
    return df

# ============================================
# MODELLTRAINING
# ============================================

def trainiere_modell():
    """
    Hauptfunktion zum Trainieren des Modells
    """
    print("="*60)
    print("🏠 IMMOBILIENPREISVORHERSAGE - MODELLTRAINING")
    print("="*60)
    
    # MLflow Experiment starten
    mlflow.set_experiment("immobilien_preisvorhersage")
    
    with mlflow.start_run(run_name=f"training_{datetime.now().strftime('%Y%m%d_%H%M')}"):
        
        # 1. Daten generieren
        print("\n📊 Generiere Trainingsdaten...")
        df = generiere_daten(anzahl=2000)
        print(f"   → {len(df)} Datensätze generiert")
        
        # 2. Feature Engineering
        print("\n🔧 Feature Engineering...")
        df_features = erstelle_features(df)
        
        # 3. Features und Zielvariable trennen
        feature_cols = [col for col in df_features.columns if col != 'preis' and col != 'preis_pro_m2']
        X = df_features[feature_cols]
        y = df_features['preis']
        
        print(f"   → {len(feature_cols)} Features: {feature_cols[:5]}...")
        
        # 4. Train-Test-Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        print(f"   → Trainingsdaten: {len(X_train)}")
        print(f"   → Testdaten: {len(X_test)}")
        
        # 5. Skalierung
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # 6. Modell training
        print("\n🤖 Trainiere Random Forest Modell...")
        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train_scaled, y_train)
        
        # 7. Evaluation
        print("\n📈 Evaluierung...")
        y_pred_train = model.predict(X_train_scaled)
        y_pred_test = model.predict(X_test_scaled)
        
        # Metriken
        train_r2 = r2_score(y_train, y_pred_train)
        test_r2 = r2_score(y_test, y_pred_test)
        test_mae = mean_absolute_error(y_test, y_pred_test)
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
        
        print(f"\n{'='*40}")
        print("ERGEBNISSE:")
        print(f"{'='*40}")
        print(f"Train R²:     {train_r2:.4f}")
        print(f"Test R²:      {test_r2:.4f}")
        print(f"Test MAE:     {test_mae:,.0f} €")
        print(f"Test RMSE:    {test_rmse:,.0f} €")
        print(f"{'='*40}")
        
        # 8. MLflow Logging
        mlflow.log_params({
            "n_estimators": 100,
            "max_depth": 10,
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "n_features": len(feature_cols)
        })
        
        mlflow.log_metrics({
            "train_r2": train_r2,
            "test_r2": test_r2,
            "test_mae": test_mae,
            "test_rmse": test_rmse
        })
        
        # 9. Feature Importance
        feature_importance = dict(zip(feature_cols, model.feature_importances_))
        wichtigste = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:5]
        print("\n🔝 Wichtigste Features:")
        for feat, imp in wichtigste:
            print(f"   {feat}: {imp:.3f}")
        
        # 10. Modelle speichern
        print("\n💾 Speichere Modelle...")
        os.makedirs("modelle", exist_ok=True)
        
        joblib.dump(model, "modelle/random_forest_model.pkl")
        joblib.dump(scaler, "modelle/scaler.pkl")
        joblib.dump(feature_cols, "modelle/feature_cols.pkl")
        
        mlflow.sklearn.log_model(model, "modell")
        
        print("   ✅ Modell gespeichert: modelle/random_forest_model.pkl")
        print("   ✅ Scaler gespeichert: modelle/scaler.pkl")
        print("   ✅ Features gespeichert: modelle/feature_cols.pkl")
        
        print("\n✨ Training erfolgreich abgeschlossen!")
        
        return model, scaler, feature_cols

# ============================================
# HAUPTFUNKTION
# ============================================

if __name__ == "__main__":
    model, scaler, features = trainiere_modell()
    
    # Beispielvorhersage
    print("\n🔍 Beispielvorhersage:")
    beispiel = pd.DataFrame([{
        'flaeche': 85,
        'zimmer': 4,
        'schlafzimmer': 2,
        'etage': 2,
        'stadt_Berlin': 1,
        'stadt_Frankfurt': 0,
        'stadt_Hamburg': 0,
        'stadt_Köln': 0,
        'stadt_München': 0,
        'stadt_Stuttgart': 0,
        'plz': 10115,
        'baujahr': 2010,
        'entfernung_bahn': 0.5,
        'garten': 1,
        'balkon': 1,
        'alter': 14,
        'zimmer_pro_m2': 4.7,
        'komfort_score': 3
    }])
    
    # Sicherstellen, dass alle Spalten vorhanden sind
    for col in features:
        if col not in beispiel.columns:
            beispiel[col] = 0
    
    beispiel = beispiel[features]
    beispiel_scaled = scaler.transform(beispiel)
    vorhersage = model.predict(beispiel_scaled)[0]
    
    print(f"   Geschätzter Preis: {vorhersage:,.0f} €")