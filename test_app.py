import pytest
from app import app

@pytest.fixture
def client():
    with app.test_client() as client:
        yield client

def test_home_page(client):
    """Test if the home page loads correctly."""
    response = client.get('/')
    assert response.status_code == 200
    assert b'Enter Text for Prediction' in response.data

def test_predict_with_valid_input(client):
    """Test prediction with valid input."""
    data = {
        'user_text': 'I love this product!',
        'model_number': '1'
    }
    response = client.post('/predict', data=data)
    assert response.status_code == 200
    assert b'Prediction:' in response.data

def test_predict_with_missing_text(client):
    """Test prediction with missing text input."""
    data = {
        'user_text': '',
        'model_number': '1'
    }
    response = client.post('/predict', data=data)
    assert response.status_code == 200
    assert b'Please enter valid text.' in response.data

def test_predict_with_invalid_model(client):
    """Test prediction with invalid model selection."""
    data = {
        'user_text': 'I love this product!',
        'model_number': '100'
    }
    response = client.post('/predict', data=data)
    assert response.status_code == 200
    assert b'Invalid model selected.' in response.data
