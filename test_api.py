# test_api.py
import pytest
import httpx
import json
from main import app
import asyncio

@pytest.fixture
def test_routine_request():
    return {
        "peak_mode": "Physique",
        "sleep_chronotype": "Third Bird",
        "nature_of_commutes": "High Commute with long driving periods",
        "nature_of_traveling": "Less than 4 times per year",
        "easiness_of_regimen": 7,
        "observation_level": 7,
        "challenges": [
            "Low nutrition IQ", 
            "prolonged sedentary periods", 
            "midday sluggishness", 
            "routine is hard to maintain"
        ]
    }

@pytest.fixture
def test_nutrition_request():
    return {
        "current_body_composition": {
            "weight": 270,
            "body_fat_percentage": 30.7,
            "lean_body_mass": 187.0
        },
        "target_goal": "Recomposition",
        "activity_level": "Moderately Active",
        "daily_step_count": 8000,
        "resting_metabolic_rate": 2203,
        "macronutrient_preferences": "Balanced",
        "intermittent_fasting": False,
        "num_meals_per_day": 4,
        "eating_window": "9am to 8pm",
        "protein_source_preference": "Mixed",
        "carb_tolerance": "Lower carb diet",
        "fat_preference": "None",
        "food_sensitivities": [],
        "cultural_dietary_category": "None",
        "cooking_ability": "No ability. Less than 6hrs/wk.",
        "meal_prep_frequency": "Daily",
        "eating_out_frequency": "Never",
        "kitchen_access": "Full kitchen",
        "existing_supplements": [
            "Protein Powder", 
            "Creatine", 
            "Ageless Male Max", 
            "Fish Oil", 
            "Zinc"
        ],
        "willingness_to_supplement": True,
        "water_intake_target": "3.5L",
        "caffeine_consumption": "Low",
        "electrolyte_needs": "Indulgent salt intake, Minimum sweating"
    }

# API endpoint testing with mocked LLM responses
@pytest.mark.asyncio
async def test_health_endpoint():
    async with httpx.AsyncClient(app=app, base_url="http://test") as client:
        response = await client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"

@pytest.mark.asyncio
async def test_routine_endpoint(test_routine_request, monkeypatch):
    # Mock LLM API call
    async def mock_call_llm_api(input_data, endpoint):
        return {
            "output": "Mocked routine plan response",
            "model": "gpt-4-test",
            "timestamp": 1234567890.0
        }
    
    # Apply the monkeypatch
    from main import call_llm_api
    monkeypatch.setattr("main.call_llm_api", mock_call_llm_api)
    
    async with httpx.AsyncClient(app=app, base_url="http://test") as client:
        response = await client.post("/api/routine", json=test_routine_request)
        assert response.status_code == 200
        assert "output" in response.json()
        assert response.json()["output"] == "Mocked routine plan response"

@pytest.mark.asyncio
async def test_nutrition_endpoint(test_nutrition_request, monkeypatch):
    # Mock LLM API call
    async def mock_call_llm_api(input_data, endpoint):
        return {
            "output": "Mocked nutrition plan response",
            "model": "gpt-4-test",
            "timestamp": 1234567890.0
        }
    
    # Apply the monkeypatch
    from main import call_llm_api
    monkeypatch.setattr("main.call_llm_api", mock_call_llm_api)
    
    async with httpx.AsyncClient(app=app, base_url="http://test") as client:
        response = await client.post("/api/nutrition", json=test_nutrition_request)
        assert response.status_code == 200
        assert "output" in response.json()
        assert response.json()["output"] == "Mocked nutrition plan response"

# Cache system testing
@pytest.mark.asyncio
async def test_caching_system(test_routine_request, monkeypatch):
    # Keep track of LLM API calls
    call_count = 0
    
    async def mock_call_llm_api(input_data, endpoint):
        nonlocal call_count
        call_count += 1
        return {
            "output": f"Mocked response (call {call_count})",
            "model": "gpt-4-test",
            "timestamp": 1234567890.0
        }
    
    # Mock cache functions
    cache_storage = {}
    
    async def mock_check_cache(cache_key):
        return cache_storage.get(cache_key)
    
    async def mock_save_to_cache(cache_key, response):
        cache_storage[cache_key] = response
    
    # Apply the monkepatches
    monkeypatch.setattr("main.call_llm_api", mock_call_llm_api)
    monkeypatch.setattr("main.check_cache", mock_check_cache)
    monkeypatch.setattr("main.save_to_cache", mock_save_to_cache)
    
    async with httpx.AsyncClient(app=app, base_url="http://test") as client:
        # First call should hit the LLM API
        response1 = await client.post("/api/routine", json=test_routine_request)
        assert response1.status_code == 200
        assert call_count == 1
        
        # Second call with same parameters should use cache
        response2 = await client.post("/api/routine", json=test_routine_request)
        assert response2.status_code == 200
        assert call_count == 1  # Call count should not increase
        assert response2.json() == response1.json()

# Performance testing
@pytest.mark.asyncio
async def test_response_time(test_routine_request, monkeypatch):
    async def mock_call_llm_api(input_data, endpoint):
        await asyncio.sleep(0.1)  # Simulate API delay
        return {
            "output": "Mocked response",
            "model": "gpt-4-test",
            "timestamp": 1234567890.0
        }
    
    # Clear cache for this test
    async def mock_check_cache(cache_key):
        return None
    
    async def mock_save_to_cache(cache_key, response):
        pass
    
    # Apply the monkepatches
    monkeypatch.setattr("main.call_llm_api", mock_call_llm_api)
    monkeypatch.setattr("main.check_cache", mock_check_cache)
    monkeypatch.setattr("main.save_to_cache", mock_save_to_cache)
    
    async with httpx.AsyncClient(app=app, base_url="http://test") as client:
        start_time = asyncio.get_event_loop().time()
        response = await client.post("/api/routine", json=test_routine_request)
        end_time = asyncio.get_event_loop().time()
        
        assert response.status_code == 200
        assert end_time - start_time < 0.5  # Test should complete in less than 500ms

# Run tests
if __name__ == "__main__":
    pytest.main(["-xvs", "test_api.py"])