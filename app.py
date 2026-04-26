from flask import Flask, request, jsonify, send_from_directory
import joblib, json, numpy as np

app  = Flask(__name__, static_folder='static')
meta = json.load(open('model_meta.json'))

FEATURE_COLS = meta['feature_cols']
ZERO_COLS    = meta['zero_cols']

# Load all 4 trained pipelines
# Each pipeline = KNNImputer → RobustScaler → Classifier
pipelines = {
    "Gradient Boosting":   joblib.load('model_gradient_boosting.pkl'),
    "Random Forest":       joblib.load('model_random_forest.pkl'),
    "Logistic Regression": joblib.load('model_logistic_regression.pkl'),
    "KNN Classifier":      joblib.load('model_knn_classifier.pkl'),
}
print("✅ All 4 models loaded successfully")

def preprocess(raw: dict) -> np.ndarray:
    """Convert raw form input into the 13-feature vector the models expect.
       Fake zeros are replaced with NaN — the KNNImputer inside
       each pipeline will fill them before prediction."""
    G   = raw['Glucose']       if raw['Glucose']       != 0 else np.nan
    BP  = raw['BloodPressure'] if raw['BloodPressure']  != 0 else np.nan
    ST  = raw['SkinThickness'] if raw['SkinThickness']  != 0 else np.nan
    INS = raw['Insulin']       if raw['Insulin']        != 0 else np.nan
    BMI = raw['BMI']           if raw['BMI']            != 0 else np.nan
    AGE = raw['Age']
    PRG = raw['Pregnancies']
    DPF = raw['DiabetesPedigreeFunction']

    # Use median defaults for engineered features when values missing
    G_  = G   if not (isinstance(G,  float) and np.isnan(G))   else 120.0
    B_  = BMI if not (isinstance(BMI,float) and np.isnan(BMI)) else 32.0
    I_  = INS if not (isinstance(INS,float) and np.isnan(INS)) else 80.0

    row = [
        PRG, G, BP, ST, INS, BMI, DPF, AGE,   # original 8 features
        G_  * B_,                               # Glucose_BMI
        AGE * PRG,                              # Age_Pregnancies
        I_  / (G_ + 1),                        # Insulin_Glucose
        B_  * AGE,                             # BMI_Age
        float(min(3, max(0, int(np.searchsorted([70,100,125], G_))))),  # Glucose_category
    ]
    return np.array([row])

def risk_label(prob):
    if prob < 30:  return 'Low'
    if prob < 55:  return 'Moderate'
    if prob < 75:  return 'High'
    return 'Very High'

@app.route('/')
def index():
    return send_from_directory('static', 'index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data       = request.get_json(force=True)
        X          = preprocess(data)
        model_name = data.get('model', meta['best_model'])
        if model_name not in pipelines:
            model_name = meta['best_model']

        pipe = pipelines[model_name]
        prob = float(pipe.predict_proba(X)[0, 1]) * 100

        # Get probabilities from all 4 models for comparison
        all_probs = {
            name: round(float(p.predict_proba(X)[0, 1]) * 100, 1)
            for name, p in pipelines.items()
        }

        return jsonify({
            'probability':   round(prob, 1),
            'prediction':    int(prob >= 50),
            'risk_level':    risk_label(prob),
            'model_used':    model_name,
            'all_probs':     all_probs,
            'model_results': meta['model_results'],
            'best_model':    meta['best_model'],
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/meta')
def get_meta():
    return jsonify(meta)

if __name__ == '__main__':
    print("\nGlycoSense running → http://localhost:5000\n")
    app.run(debug=True, port=5000)
