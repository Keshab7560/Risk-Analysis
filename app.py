from flask import Flask, request, render_template, jsonify, make_response
import numpy as np
import joblib
from datetime import datetime
from fpdf import FPDF
import io
import traceback
import os

app = Flask(__name__)

# Configure upload folder for potential logo
app.config['UPLOAD_FOLDER'] = 'static'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load the model and scaler
try:
    model = joblib.load("health_risk_model.pkl")
    scaler = joblib.load("scaler.pkl")
    print("Model and scaler loaded successfully")
except Exception as e:
    print(f"Error loading model: {str(e)}")
    model = None
    scaler = None

# Map predictions to risk categories
risk_labels = {
    0: "High Risk",
    1: "Low Risk", 
    2: "Medium Risk"
}

class HealthReportPDF(FPDF):
    def __init__(self):
        super().__init__()
        self.set_auto_page_break(auto=True, margin=15)
        
    def header(self):
        # Logo - replace with your logo path if available
        logo_path = os.path.join(app.config['UPLOAD_FOLDER'], 'logo.png')
        if os.path.exists(logo_path):
            self.image(logo_path, 10, 8, 25)
        
        self.set_font('Arial', 'B', 16)
        self.cell(0, 10, 'Health Risk Assessment Report', 0, 1, 'C')
        self.ln(10)
    
    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')
    
    def add_section(self, title, data, is_dict=True):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, title, 0, 1)
        self.set_font('Arial', '', 10)
        
        if is_dict:
            for key, value in data.items():
                self.cell(40, 10, key, 0, 0)
                self.multi_cell(0, 10, str(value), 0, 1)
        else:
            for item in data:
                self.multi_cell(0, 10, item, 0, 1)
        
        self.ln(5)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if not model or not scaler:
            return jsonify({
                'error': 'Prediction model not loaded',
                'status': 'error'
            }), 500

        # Get form data
        data = request.get_json()
        
        # Validate required fields
        required_fields = ['age', 'gender', 'bmi', 'bloodPressure', 'cholesterol', 'glucose']
        if not all(field in data for field in required_fields):
            return jsonify({
                'error': 'Missing required fields',
                'status': 'error'
            }), 400

        # Extract and validate features
        try:
            features = [
                float(data['age']),
                float(data['gender']),
                float(data['bmi']),
                float(data['bloodPressure']),
                float(data['cholesterol']),
                float(data['glucose'])
            ]
        except ValueError as e:
            return jsonify({
                'error': f'Invalid numeric value: {str(e)}',
                'status': 'error'
            }), 400

        # Scale features
        final_features = scaler.transform([features])
        
        # Make prediction
        prediction = model.predict(final_features)
        predicted_class = int(np.clip(np.round(prediction[0]), 0, 2))
        risk_category = risk_labels[predicted_class]
        
        # Calculate risk percentage
        risk_score = calculate_risk_score(features)
        
        # Prepare response
        response = {
            'status': 'success',
            'prediction': risk_category,
            'risk_score': f"{risk_score:.1f}%",
            'report_data': {
                'patient_name': data.get('patientName', 'Anonymous'),
                'age': features[0],
                'gender': 'Male' if features[1] == 0 else 'Female',
                'bmi': features[2],
                'blood_pressure': features[3],
                'cholesterol': features[4],
                'glucose': features[5],
                'risk_category': risk_category,
                'risk_score': f"{risk_score:.1f}%",
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
        }
        
        return jsonify(response)

    except Exception as e:
        app.logger.error(f"Prediction failed: {str(e)}\n{traceback.format_exc()}")
        return jsonify({
            'error': 'Internal server error',
            'status': 'error'
        }), 500

@app.route('/generate_pdf', methods=['POST'])
def generate_pdf():
    try:
        data = request.get_json()
        
        # Validate required data
        if not data or 'patient_name' not in data:
            return jsonify({'error': 'Invalid data for PDF generation'}), 400

        # Create PDF
        pdf = HealthReportPDF()
        pdf.add_page()
        
        # Add patient information
        pdf.add_section('Patient Information', {
            'Name': data.get('patient_name', 'N/A'),
            'Age': data.get('age', 'N/A'),
            'Gender': data.get('gender', 'N/A'),
            'Report Date': data.get('timestamp', datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        })
        
        # Add health metrics
        pdf.add_section('Health Metrics', {
            'BMI': data.get('bmi', 'N/A'),
            'Blood Pressure': f"{data.get('blood_pressure', 'N/A')} mmHg",
            'Cholesterol': f"{data.get('cholesterol', 'N/A')} mg/dL",
            'Glucose': f"{data.get('glucose', 'N/A')} mg/dL"
        })
        
        # Add risk assessment with color coding
        risk_category = data.get('risk_category', 'N/A')
        risk_score = data.get('risk_score', 'N/A')
        
        if 'High' in risk_category:
            pdf.set_text_color(255, 0, 0)  # Red
        elif 'Medium' in risk_category:
            pdf.set_text_color(220, 120, 0)  # Orange
        else:
            pdf.set_text_color(0, 128, 0)  # Green
            
        pdf.add_section('Risk Assessment', {
            'Risk Category': risk_category,
            'Risk Score': risk_score
        })
        pdf.set_text_color(0, 0, 0)  # Reset color
        
        # Add recommendations
        recommendations = generate_recommendations(risk_category)
        pdf.add_section('Recommendations', recommendations, is_dict=False)
        
        # Add disclaimer
        pdf.add_section('Disclaimer', [
            'This report is generated automatically and should not replace professional medical advice.',
            'Consult your healthcare provider for interpretation of these results.'
        ], is_dict=False)
        
        # Save to memory buffer
        pdf_bytes = pdf.output(dest='S').encode('latin1')
        
        # Create response
        response = make_response(pdf_bytes)
        response.headers['Content-Type'] = 'application/pdf'
        response.headers['Content-Disposition'] = \
            f'attachment; filename=health_report_{data["patient_name"].replace(" ", "_")}.pdf'
        
        return response

    except Exception as e:
        app.logger.error(f"PDF generation failed: {str(e)}\n{traceback.format_exc()}")
        return jsonify({
            'error': 'Failed to generate PDF',
            'details': str(e)
        }), 500

def calculate_risk_score(features):
    """Calculate a simulated risk score based on input features"""
    age, gender, bmi, bp, chol, glucose = features
    
    # Calculate individual risk factors (0-1 range)
    age_factor = min(max((age - 30) / 50, 0), 1)  # 30-80 range
    bmi_factor = min(max((bmi - 18.5) / 31.5, 0), 1)  # 18.5-50 range
    bp_factor = min(max((bp - 90) / 60, 0), 1)  # 90-150 range
    chol_factor = min(max((chol - 120) / 180, 0), 1)  # 120-300 range
    glucose_factor = min(max((glucose - 70) / 130, 0), 1)  # 70-200 range
    
    # Weighted average
    risk_score = (
        age_factor * 0.25 +
        bmi_factor * 0.25 + 
        bp_factor * 0.2 +
        chol_factor * 0.15 +
        glucose_factor * 0.15
    )
    
    return min(100, max(0, risk_score * 100))

def generate_recommendations(risk_category):
    """Generate personalized recommendations based on risk category"""
    base_recommendations = [
        "Maintain a balanced diet rich in fruits and vegetables",
        "Engage in regular physical activity (150+ minutes/week)",
        "Get adequate sleep (7-9 hours per night)",
        "Manage stress through relaxation techniques",
        "Schedule regular health checkups"
    ]
    
    if risk_category == "High Risk":
        return [
            "URGENT: Consult a healthcare professional immediately",
            "Consider comprehensive medical evaluation",
            *base_recommendations,
            "Monitor your health indicators regularly",
            "Follow any prescribed treatment plans strictly"
        ]
    elif risk_category == "Medium Risk":
        return [
            "Consult your healthcare provider for personalized advice",
            "Focus on improving lifestyle factors",
            *base_recommendations,
            "Monitor your progress with regular testing"
        ]
    else:
        return [
            "Continue maintaining your healthy habits",
            *base_recommendations,
            "Stay vigilant with annual health screenings"
        ]

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)