from flask import Flask, request, jsonify, render_template
from src.pipeline.predict_pipeline import PredictPipeline, CustomData
from src.exception import CustomException
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Initialize the prediction pipeline
predict_pipeline = PredictPipeline()

def get_credit_rating(credit_score):
    """Convert credit score to credit rating"""
    if credit_score >= 800:
        return "Excellent"
    elif credit_score >= 740:
        return "Very Good"
    elif credit_score >= 670:
        return "Good"
    elif credit_score >= 580:
        return "Fair"
    else:
        return "Poor"

@app.route('/')
def home():
    """Home page with prediction form"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """API endpoint for making credit score predictions"""
    try:
        # Get JSON data from request
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Create CustomData object
        custom_data = CustomData(
            user_id=data['user_id'],
            age=int(data['age']),
            gender=data['gender'],
            education_level=data['education_level'],
            employment_status=data['employment_status'],
            job_title=data['job_title'],
            monthly_income_usd=float(data['monthly_income_usd']),
            monthly_expenses_usd=float(data['monthly_expenses_usd']),
            savings_usd=float(data['savings_usd']),
            has_loan=data['has_loan'],
            loan_type=data['loan_type'],
            loan_amount_usd=float(data['loan_amount_usd']),
            loan_term_months=int(data['loan_term_months']),
            monthly_emi_usd=float(data['monthly_emi_usd']),
            loan_interest_rate_pct=float(data['loan_interest_rate_pct']),
            debt_to_income_ratio=float(data['debt_to_income_ratio']),
            savings_to_income_ratio=float(data['savings_to_income_ratio']),
            region=data['region'],
            record_date=data['record_date']
        )
        
        # Convert to DataFrame
        features_df = custom_data.get_data_as_data_frame()
        
        # Make prediction
        prediction = predict_pipeline.predict(features_df)
        
        # Get credit score (assuming the model predicts credit score directly)
        credit_score = float(prediction[0]) if len(prediction) > 0 else 0.0
        
        # Ensure credit score is within reasonable bounds (300-850)
        credit_score = max(300, min(850, credit_score))
        
        result = {
            'credit_score': int(credit_score),
            'credit_rating': get_credit_rating(credit_score),
            'user_id': data['user_id']
        }
        
        logger.info(f"Credit score prediction made for user {data['user_id']}: {credit_score}")
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'Credit Score Prediction Service'
    })

if __name__ == '__main__':
    # Check if model files exist
    model_path = os.path.join("artifacts", "model.pkl")
    preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")
    
    if not os.path.exists(model_path):
        print(f"ERROR: Model file not found at {model_path}")
        exit(1)
    
    if not os.path.exists(preprocessor_path):
        print(f"ERROR: Preprocessor file not found at {preprocessor_path}")
        exit(1)
    
    print("Starting Credit Score Prediction Service...")
    app.run(debug=True, host='0.0.0.0', port=5000) 