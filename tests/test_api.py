import pytest
import json
from app import app

@pytest.fixture
def client():
    """Create a test client for the Flask app"""
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_health_endpoint(client):
    """Test the health check endpoint"""
    response = client.get('/health')
    assert response.status_code == 200
    data = json.loads(response.data)
    assert data['status'] == 'healthy'
    assert data['service'] == 'Credit Score Prediction Service'

def test_predict_endpoint_valid_data(client):
    """Test the prediction endpoint with valid data"""
    test_data = {
        "user_id": "TEST001",
        "age": 30,
        "gender": "Male",
        "education_level": "Bachelor's",
        "employment_status": "Full-time",
        "job_title": "Engineer",
        "monthly_income_usd": 5000.0,
        "monthly_expenses_usd": 2000.0,
        "savings_usd": 10000.0,
        "has_loan": "No",
        "loan_type": "Home",
        "loan_amount_usd": 0.0,
        "loan_term_months": 0,
        "monthly_emi_usd": 0.0,
        "loan_interest_rate_pct": 0.0,
        "debt_to_income_ratio": 0.3,
        "savings_to_income_ratio": 2.0,
        "region": "North America",
        "record_date": "2024-01-01"
    }
    
    response = client.post('/predict',
                          data=json.dumps(test_data),
                          content_type='application/json')
    
    assert response.status_code == 200
    data = json.loads(response.data)
    assert 'credit_score' in data
    assert 'credit_rating' in data
    assert 'user_id' in data
    assert isinstance(data['credit_score'], int)
    assert data['credit_score'] >= 300 and data['credit_score'] <= 850

def test_predict_endpoint_missing_data(client):
    """Test the prediction endpoint with missing data"""
    test_data = {
        "user_id": "TEST001",
        "age": 30
        # Missing required fields
    }
    
    response = client.post('/predict',
                          data=json.dumps(test_data),
                          content_type='application/json')
    
    assert response.status_code == 500  # Should handle missing fields gracefully

def test_predict_endpoint_invalid_data(client):
    """Test the prediction endpoint with invalid data"""
    test_data = {
        "user_id": "TEST001",
        "age": "invalid_age",  # Invalid age type
        "gender": "Male",
        "education_level": "Bachelor's",
        "employment_status": "Full-time",
        "job_title": "Engineer",
        "monthly_income_usd": 5000.0,
        "monthly_expenses_usd": 2000.0,
        "savings_usd": 10000.0,
        "has_loan": "No",
        "loan_type": "Home",
        "loan_amount_usd": 0.0,
        "loan_term_months": 0,
        "monthly_emi_usd": 0.0,
        "loan_interest_rate_pct": 0.0,
        "debt_to_income_ratio": 0.3,
        "savings_to_income_ratio": 2.0,
        "region": "North America",
        "record_date": "2024-01-01"
    }
    
    response = client.post('/predict',
                          data=json.dumps(test_data),
                          content_type='application/json')
    
    assert response.status_code == 500  # Should handle invalid data gracefully

def test_home_endpoint(client):
    """Test the home page endpoint"""
    response = client.get('/')
    assert response.status_code == 200
    assert b'Credit Score Prediction Service' in response.data 