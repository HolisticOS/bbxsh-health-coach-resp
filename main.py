# # main.py
# from fastapi import FastAPI, Request, HTTPException, Depends
# from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel
# import os
# import json
# import aiohttp
# import asyncio
# import time
# import firebase_admin
# from firebase_admin import credentials, firestore
# from typing import Dict, Any, Optional, List

# from dotenv import load_dotenv

# # Initialize Firebase
# try:
#     cred = credentials.Certificate("firebase-service-account.json")
#     firebase_admin.initialize_app(cred)
# except ValueError:
#     # For local development without credentials
#     firebase_admin.initialize_app()
    
# db = firestore.client()

# app = FastAPI(title="Health Coach API", 
#               description="API for health coaching chatbot",
#               version="1.0.0")

# # Add CORS middleware
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],  # Restrict in production
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# load_dotenv()

# # Environment variables
# LLM_API_URL = os.environ.get("LLM_API_URL")
# API_KEY = os.environ.get("LLM_API_KEY")

# import logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# if not LLM_API_URL:
#     logger.warning("LLM_API_URL environment variable is not set! Using OpenAI API by default.")
#     LLM_API_URL = "https://api.openai.com/v1/chat/completions"

# if not API_KEY:
#     logger.error("LLM_API_KEY environment variable is not set! API calls will fail.")
# # Cache timeout (24 hours)
# CACHE_TIMEOUT = 86400

# # Models
# class RoutineRequest(BaseModel):
#     peak_mode: str
#     sleep_chronotype: str
#     nature_of_commutes: str
#     nature_of_traveling: str
#     easiness_of_regimen: int
#     observation_level: int
#     challenges: List[str]

# class NutritionRequest(BaseModel):
#     current_body_composition: Dict[str, Any]
#     target_goal: str
#     activity_level: str
#     daily_step_count: int
#     resting_metabolic_rate: int
#     macronutrient_preferences: str
#     intermittent_fasting: bool
#     num_meals_per_day: int
#     eating_window: str
#     protein_source_preference: str
#     carb_tolerance: str
#     fat_preference: str
#     food_sensitivities: List[str]
#     cultural_dietary_category: str
#     cooking_ability: str
#     meal_prep_frequency: str
#     eating_out_frequency: str
#     kitchen_access: str
#     existing_supplements: List[str]
#     willingness_to_supplement: bool
#     water_intake_target: str
#     caffeine_consumption: str
#     electrolyte_needs: str

# # Authentication middleware
# def get_token_from_header(request: Request):
#     auth_header = request.headers.get("Authorization")
#     if auth_header and auth_header.startswith("Bearer "):
#         return auth_header.split(" ")[1]
#     return None

# # Cache utilities
# def generate_cache_key(data: dict) -> str:
#     """Generate a cache key from request data"""
#     serialized = json.dumps(data, sort_keys=True)
#     import hashlib
#     return hashlib.md5(serialized.encode()).hexdigest()

# async def check_cache(cache_key: str) -> Optional[dict]:
#     """Check if result exists in Firebase cache"""
#     cache_ref = db.collection('response_cache').document(cache_key)
#     cache_doc = cache_ref.get()
    
#     if cache_doc.exists:
#         cache_data = cache_doc.to_dict()
#         # Check if cache is still valid
#         if time.time() - cache_data['timestamp'] < CACHE_TIMEOUT:
#             return cache_data['response']
#     return None

# async def save_to_cache(cache_key: str, response: dict):
#     """Save response to Firebase cache"""
#     db.collection('response_cache').document(cache_key).set({
#         'response': response,
#         'timestamp': time.time()
#     })

# # Optimized LLM request function
# async def call_llm_api(input_data: dict, endpoint: str):
#     """Make optimized request to LLM API"""
    
#     # Construct prompt based on endpoint
#     if endpoint == "routine":
#         prompt = construct_routine_prompt(input_data)
#     elif endpoint == "nutrition":
#         prompt = construct_nutrition_prompt(input_data)
#     else:
#         raise ValueError(f"Unknown endpoint: {endpoint}")
    
    
#     # Validate credentials
#     if not API_KEY:
#         raise HTTPException(
#             status_code=500,
#             detail="LLM API key not configured. Please set the LLM_API_KEY environment variable."
#         )
    
#     # Log request (without sensitive info)
#     logger.info(f"Making {endpoint} request to {LLM_API_URL}")
    
#     # Prepare API request
#     headers = {
#         "Authorization": f"Bearer {API_KEY}",
#         "Content-Type": "application/json"
#     }
    
#     payload = {
#         "model": "gpt-4o-mini",  # Use your preferred model
#         "messages": [
#             {"role": "system", "content": "You are a professional health coach specializing in creating personalized health and nutrition plans."},
#             {"role": "user", "content": prompt}
#         ],
#         "temperature": 0.3,  # Lower temperature for more consistent outputs
#         "top_p": 1,
#         "stream": False  # Set to True if you want to implement streaming
#     }
    
#     # Corrected indentation for the try block
#     try:
#         # Use aiohttp for async requests
#         async with aiohttp.ClientSession() as session:
#             async with session.post(LLM_API_URL, json=payload, headers=headers) as response:
#                 if response.status == 200:
#                     result = await response.json()
#                     logger.info(f"Successful LLM API response for {endpoint}")
#                     return {
#                         "output": result["choices"][0]["message"]["content"],
#                         "model": result["model"],
#                         "timestamp": time.time()
#                     }
#                 else:
#                     error_text = await response.text()
#                     logger.error(f"LLM API error: Status {response.status}, Response: {error_text}")
#                     raise HTTPException(
#                         status_code=response.status, 
#                         detail=f"LLM API error: {error_text}"
#                     )
#     except aiohttp.ClientError as e:
#         logger.error(f"Network error calling LLM API: {str(e)}")
#         raise HTTPException(
#             status_code=503,
#             detail=f"Failed to connect to LLM API: {str(e)}"
#         )
#     except Exception as e:
#         logger.error(f"Unexpected error in LLM API call: {str(e)}")
#         raise HTTPException(
#             status_code=500,
#             detail=f"Unexpected error: {str(e)}"
#         )
# # Prompt construction functions
# def construct_routine_prompt(data: dict) -> str:
#     """Construct an optimized prompt for routine planning with detailed coach instructions"""
#     return f"""
# You are a master health coach with deep expertise in nutrition science, chronobiology, and cognitive behavioral science. 
# Your mission is to help users optimize their daily routines and lifestyle choices based on their natural energy patterns and specific goals. 
# When users provide their 7 key inputs, create a structured daily routine following this exact format:

# Here's a tailored routine to help you optimize {data['peak_mode']} while addressing your challenges:

# Pre-First Wind (Upon Waking Up):
# * [Recommendation 1]: [Brief description]
# * [Recommendation 2]: [Brief description]
# * [Recommendation 3]: [Brief description]

# First Wind (Morning Work Session):
# * [Recommendation 1]: [Brief description]
# * [Recommendation 2]: [Brief description]
# * [Recommendation 3]: [Brief description]

# Midday Slump (Lunch Break):
# * [Recommendation 1]: [Brief description]
# * [Recommendation 2]: [Brief description]
# * [Recommendation 3]: [Brief description]

# Pre-Second Wind (Afternoon Work Session):
# * [Recommendation 1]: [Brief description]
# * [Recommendation 2]: [Brief description]
# * [Recommendation 3]: [Brief description]

# Second Wind (Evening Routine):
# * [Recommendation 1]: [Brief description]
# * [Recommendation 2]: [Brief description]
# * [Recommendation 3]: [Brief description]

# Unwind (Before Bed):
# * [Recommendation 1]: [Brief description]
# * [Recommendation 2]: [Brief description]
# * [Recommendation 3]: [Brief description]

# By following this routine, you'll enhance your {data['peak_mode']} while addressing challenges: {', '.join(data['challenges'])}.

# User's Inputs:
# 1. Peak Mode: {data['peak_mode']}
# 2. Sleep Chronotype: {data['sleep_chronotype']}
# 3. Nature of Commutes: {data['nature_of_commutes']}
# 4. Nature of Traveling: {data['nature_of_traveling']}
# 5. Easiness of Regimen: {data['easiness_of_regimen']}
# 6. Observation Level: {data['observation_level']}
# 7. Challenges: {', '.join(data['challenges'])}

# Create personalized, actionable, and concise recommendations addressing the user's specific needs.
# """


# def construct_nutrition_prompt(data: dict) -> str:
#     """Construct an optimized prompt for nutrition planning"""
#     return f"""
#     Please create a comprehensive nutrition plan for a client with the following parameters:
    
#     1. BASELINE METRICS & GOALS:
#        - Current Body Composition: {data['current_body_composition']}
#        - Target Goal: {data['target_goal']}
#        - Activity Level: {data['activity_level']}
#        - Daily Step Count: {data['daily_step_count']}
    
#     2. METABOLIC & ENERGY NEEDS:
#        - Resting Metabolic Rate: {data['resting_metabolic_rate']}
#        - Macronutrient Preferences: {data['macronutrient_preferences']}
    
#     3. MEAL TIMING & FASTING:
#        - Intermittent Fasting: {data['intermittent_fasting']}
#        - Number of Meals per Day: {data['num_meals_per_day']}
#        - Eating Window: {data['eating_window']}
    
#     4. DIETARY PREFERENCES:
#        - Protein Source: {data['protein_source_preference']}
#        - Carb Tolerance: {data['carb_tolerance']}
#        - Fat Preference: {data['fat_preference']}
#        - Food Sensitivities: {', '.join(data['food_sensitivities']) if data['food_sensitivities'] else 'None'}
#        - Cultural Dietary Category: {data['cultural_dietary_category']}
    
#     5. MEAL PREPARATION:
#        - Cooking Ability: {data['cooking_ability']}
#        - Meal Prep Frequency: {data['meal_prep_frequency']}
#        - Eating Out Frequency: {data['eating_out_frequency']}
#        - Kitchen Access: {data['kitchen_access']}
    
#     6. SUPPLEMENTS:
#        - Existing Supplements: {', '.join(data['existing_supplements'])}
#        - Willing to Add Supplements: {data['willingness_to_supplement']}
    
#     7. HYDRATION:
#        - Water Intake Target: {data['water_intake_target']}
#        - Caffeine Consumption: {data['caffeine_consumption']}
#        - Electrolyte Needs: {data['electrolyte_needs']}
    
#     Provide a detailed nutrition plan including:
#     1. Daily caloric target and macronutrient breakdown
#     2. Meal-by-meal breakdown with portion sizes
#     3. Specific food recommendations and alternatives
#     4. Supplement recommendations
#     5. Hydration schedule
#     6. Sample 5-day meal plan
#     Format the response with clear headers and bullet points for easy reading.
#     """

# # API endpoints
# @app.post("/api/routine", response_description="Generate a personalized routine plan")
# async def generate_routine(request: RoutineRequest, client_request: Request):
#     """Generate a routine plan based on user parameters"""
#     # Generate cache key from request data
#     cache_key = generate_cache_key(request.dict())
    
#     # Check cache first
#     cached_response = await check_cache(cache_key)
#     if cached_response:
#         return cached_response
    
#     # Make request to LLM API
#     try:
#         response = await call_llm_api(request.dict(), "routine")
        
#         # Save to cache
#         await save_to_cache(cache_key, response)
        
#         return response
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

# @app.post("/api/nutrition", response_description="Generate a personalized nutrition plan")
# async def generate_nutrition_plan(request: NutritionRequest, client_request: Request):
#     """Generate a nutrition plan based on user parameters"""
#     # Generate cache key from request data
#     cache_key = generate_cache_key(request.dict())
    
#     # Check cache first
#     cached_response = await check_cache(cache_key)
#     if cached_response:
#         return cached_response
    
#     # Make request to LLM API
#     try:
#         response = await call_llm_api(request.dict(), "nutrition")
        
#         # Save to cache
#         await save_to_cache(cache_key, response)
        
#         return response
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

# @app.get("/health", response_description="Check API health")
# async def health_check():
#     """Health check endpoint"""
#     return {"status": "healthy", "service": "health-coach-api"}

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)


# main.py with enhanced prompt construction
from fastapi import FastAPI, Request, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import json
import aiohttp
import asyncio
import time
import firebase_admin
from firebase_admin import credentials, firestore
from typing import Dict, Any, Optional, List

from dotenv import load_dotenv

# Initialize Firebase
try:
    cred = credentials.Certificate("firebase-service-account.json")
    firebase_admin.initialize_app(cred)
except ValueError:
    # For local development without credentials
    firebase_admin.initialize_app()
    
db = firestore.client()

app = FastAPI(title="Health Coach API", 
              description="API for health coaching chatbot",
              version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://bbxsh-93b83.web.app"],  # Only allow your frontend domain
    # allow_origins=["https://bbxsh-93b83.web.app", "http://localhost:51295", "http://127.0.0.1:8000"],
    allow_credentials=True,
    allow_methods=["*"],  # You can restrict methods like ["POST", "GET"] if needed
    allow_headers=["*"],  # You can restrict specific headers
)


load_dotenv()

# Environment variables
LLM_API_URL = os.environ.get("LLM_API_URL")
API_KEY = os.environ.get("LLM_API_KEY")

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if not LLM_API_URL:
    logger.warning("LLM_API_URL environment variable is not set! Using OpenAI API by default.")
    LLM_API_URL = "https://api.openai.com/v1/chat/completions"

if not API_KEY:
    logger.error("LLM_API_KEY environment variable is not set! API calls will fail.")
# Cache timeout (24 hours)
CACHE_TIMEOUT = 86400

# Models
class RoutineRequest(BaseModel):
    peak_mode: str
    sleep_chronotype: str
    nature_of_commutes: str
    nature_of_traveling: str
    easiness_of_regimen: int
    observation_level: int
    challenges: List[str]

class NutritionRequest(BaseModel):
    current_body_composition: Dict[str, Any]
    target_goal: str
    activity_level: str
    daily_step_count: int
    resting_metabolic_rate: int
    macronutrient_preferences: str
    intermittent_fasting: bool
    num_meals_per_day: int
    eating_window: str
    protein_source_preference: str
    carb_tolerance: str
    fat_preference: str
    food_sensitivities: List[str]
    cultural_dietary_category: str
    cooking_ability: str
    meal_prep_frequency: str
    eating_out_frequency: str
    kitchen_access: str
    existing_supplements: List[str]
    willingness_to_supplement: bool
    water_intake_target: str
    caffeine_consumption: str
    electrolyte_needs: str

# Also add a new model for follow-up questions (optional for implementation)
class FollowUpRequest(BaseModel):
    user_id: str
    original_request_type: str  # "routine" or "nutrition"
    original_request_data: Dict[str, Any]
    follow_up_question: str

# Authentication middleware
def get_token_from_header(request: Request):
    auth_header = request.headers.get("Authorization")
    if auth_header and auth_header.startswith("Bearer "):
        return auth_header.split(" ")[1]
    return None

# Cache utilities
def generate_cache_key(data: dict) -> str:
    """Generate a cache key from request data"""
    serialized = json.dumps(data, sort_keys=True)
    import hashlib
    return hashlib.md5(serialized.encode()).hexdigest()

async def check_cache(cache_key: str) -> Optional[dict]:
    """Check if result exists in Firebase cache"""
    cache_ref = db.collection('response_cache').document(cache_key)
    cache_doc = cache_ref.get()
    
    if cache_doc.exists:
        cache_data = cache_doc.to_dict()
        # Check if cache is still valid
        if time.time() - cache_data['timestamp'] < CACHE_TIMEOUT:
            return cache_data['response']
    return None

async def save_to_cache(cache_key: str, response: dict):
    """Save response to Firebase cache"""
    db.collection('response_cache').document(cache_key).set({
        'response': response,
        'timestamp': time.time()
    })

# Optimized LLM request function
async def call_llm_api(input_data: dict, endpoint: str):
    """Make optimized request to LLM API"""
    
    # Construct prompt based on endpoint
    if endpoint == "routine":
        prompt = construct_routine_prompt(input_data)
    elif endpoint == "nutrition":
        prompt = construct_nutrition_prompt(input_data)
    elif endpoint == "follow_up":
        prompt = construct_follow_up_prompt(input_data)
    elif endpoint == "chat":
        prompt = construct_chat_prompt(input_data)
    else:
        raise ValueError(f"Unknown endpoint: {endpoint}")
    
    # Validate credentials
    if not API_KEY:
        raise HTTPException(
            status_code=500,
            detail="LLM API key not configured. Please set the LLM_API_KEY environment variable."
        )
    
    # Log request (without sensitive info)
    logger.info(f"Making {endpoint} request to {LLM_API_URL}")
    
    # Prepare API request
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    
    # Use a more conversational system prompt
    system_prompt = """You are a master health coach with deep expertise in nutrition science, chronobiology, and cognitive behavioral science. 
    Your communication style is warm, personalized, and conversational, as if you're speaking directly to your client. 
    You explain the reasoning behind your recommendations and provide specific, actionable choices tailored to the client's lifestyle.
    You always structure your responses by time periods of the day, connecting recommendations to the client's energy patterns."""
    
    payload = {
        "model": "gpt-4o",  # Using a more capable model
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.5,  # Higher temperature for more conversational responses
        "top_p": 1,
        "stream": False  # Set to True if you want to implement streaming
    }
    
    try:
        # Use aiohttp for async requests
        async with aiohttp.ClientSession() as session:
            async with session.post(LLM_API_URL, json=payload, headers=headers) as response:
                if response.status == 200:
                    result = await response.json()
                    logger.info(f"Successful LLM API response for {endpoint}")
                    return {
                        "output": result["choices"][0]["message"]["content"],
                        "model": result["model"],
                        "timestamp": time.time()
                    }
                else:
                    error_text = await response.text()
                    logger.error(f"LLM API error: Status {response.status}, Response: {error_text}")
                    raise HTTPException(
                        status_code=response.status, 
                        detail=f"LLM API error: {error_text}"
                    )
    except aiohttp.ClientError as e:
        logger.error(f"Network error calling LLM API: {str(e)}")
        raise HTTPException(
            status_code=503,
            detail=f"Failed to connect to LLM API: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Unexpected error in LLM API call: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Unexpected error: {str(e)}"
        )

# Improved prompt construction functions
def construct_routine_prompt(data: dict) -> str:
    """Construct a conversational prompt for routine planning"""
    return f"""
Create a personalized daily routine for a client with the following preferences:

Peak Mode: {data['peak_mode']}
Sleep Chronotype: {data['sleep_chronotype']}
Nature of Commutes: {data['nature_of_commutes']}
Nature of Traveling: {data['nature_of_traveling']}
Easiness of Regimen: {data['easiness_of_regimen']} (1-10, where 10 is easiest)
Observation Level: {data['observation_level']} (1-10, where 10 is highest)
Challenges: {', '.join(data['challenges'])}

Present your response in a conversational, personalized format structured by time periods of the day:

Pre-First Wind (Upon Waking Up):
First Wind (Morning Work Session):
Midday Slump (Lunch Break):
Pre-Second Wind (Afternoon Work Session):
Second Wind (Evening Routine):
Unwind (Before Bed):

For each time period:
1. Explain the significance of this energy phase
2. Provide 2-3 specific recommendations with brief explanations
3. Connect recommendations directly to their challenges and peak mode
4. Use a warm, conversational tone as if speaking directly to the client

End with a brief summary of how following this routine will help address their specific challenges and support their peak mode goal.
"""

def construct_nutrition_prompt(data: dict) -> str:
    """Construct a conversational prompt for nutrition planning by time period"""
    challenges_text = ""
    if hasattr(data, 'challenges') and data['challenges']:
        challenges_text = f"Challenges: {', '.join(data['challenges'])}"
    
    # Determine peak mode if available
    peak_mode = data.get('peak_mode', 'Physique')
    
    return f"""
Create a personalized nutrition plan for a client with the following parameters:

Target Goal: {data['target_goal']}
Current Body Composition: {data['current_body_composition']}
Activity Level: {data['activity_level']}
Daily Step Count: {data['daily_step_count']}
Resting Metabolic Rate: {data['resting_metabolic_rate']}
Macronutrient Preferences: {data['macronutrient_preferences']}
Protein Source Preference: {data['protein_source_preference']}
Carb Tolerance: {data['carb_tolerance']}
Fat Preference: {data['fat_preference']}
Food Sensitivities: {', '.join(data['food_sensitivities']) if data['food_sensitivities'] else 'None'}
Existing Supplements: {', '.join(data['existing_supplements'])}
Water Intake Target: {data['water_intake_target']}
Peak Mode: {peak_mode}
{challenges_text}

Instead of presenting this as a structured nutrition document, respond in a personalized, conversational format organized by time periods of the day. Format your response like this:

Based on your {peak_mode} focus and your profile, here's a personalized nutrition plan that aligns with your daily routine:

Pre-First Wind (Upon Waking Up) - This is the time to gently awaken your body:
[Nutrition Strategy]: [Brief explanation connecting to their physiology and goals]
* Choice 1: [Specific food/meal option with details]
* Choice 2: [Alternative option with details]
This approach helps with [specific goal or challenge] by [explanation of benefits].

First Wind (Morning Work Session) - When your energy begins to rise:
[Continue this pattern for each time period]

Include a brief note about hydration and supplementation that integrates with their daily routine and supports their specific goals.

Your response should feel like personalized coaching advice rather than a clinical nutrition document.
"""

def construct_follow_up_prompt(data: dict) -> str:
    """Construct a prompt for follow-up questions maintaining continuity"""
    original_type = data['original_request_type']
    original_data = data['original_request_data']
    follow_up_question = data['follow_up_question']
    
    # Base context on original request type
    if original_type == "routine":
        context = f"""
        The client previously received a daily routine recommendation with these parameters:
        Peak Mode: {original_data.get('peak_mode')}
        Sleep Chronotype: {original_data.get('sleep_chronotype')}
        Challenges: {', '.join(original_data.get('challenges', []))}
        """
    else:  # nutrition
        context = f"""
        The client previously received a nutrition plan with these parameters:
        Target Goal: {original_data.get('target_goal')}
        Macronutrient Preferences: {original_data.get('macronutrient_preferences')}
        Carb Tolerance: {original_data.get('carb_tolerance')}
        Protein Source: {original_data.get('protein_source_preference')}
        """
    
    return f"""
    {context}
    
    Now they're asking: "{follow_up_question}"
    
    Respond in the same conversational, personalized format organized by time periods of the day.
    Ensure your response directly addresses their question while maintaining continuity with their previous plan.
    Keep the warm, friendly tone and provide specific, actionable recommendations with explanations of the benefits.
    """
    
class ChatRequest(BaseModel):
    message: str
    user_id: Optional[str] = "anonymous"
    conversation_history: Optional[List[Dict[str, Any]]] = []

# Function to construct chat prompts
def construct_chat_prompt(data: dict) -> str:
    """Construct a prompt for chat interactions"""
    message = data['message']
    conversation_history = data.get('conversation_history', [])
    
    # Get user context if available
    user_id = data.get('user_id', 'anonymous')
    user_doc = db.collection('user_requests').document(user_id).get()
    user_context = ""
    
    if user_doc.exists:
        user_data = user_doc.to_dict()
        if 'last_routine_request' in user_data:
            routine = user_data['last_routine_request']
            user_context += f"""
            The user previously created a routine plan with peak mode: {routine.get('peak_mode')}
            and identified challenges: {', '.join(routine.get('challenges', []))}
            """
        
        if 'last_nutrition_request' in user_data:
            nutrition = user_data['last_nutrition_request']
            user_context += f"""
            The user previously created a nutrition plan with target goal: {nutrition.get('target_goal')}
            and macronutrient preferences: {nutrition.get('macronutrient_preferences')}
            """
    
    return f"""
    {user_context}
    
    The user has sent the following message: "{message}"
    
    Respond as a health coach in a warm, conversational tone. Be helpful and supportive while providing actionable advice.
    
    If they're asking about creating a detailed routine or nutrition plan, suggest they use the dedicated forms in the app for the best personalized experience.
    
    If they have a specific health or nutrition question, provide thoughtful guidance based on your expertise, while being careful not to make medical claims.
    
    If they mention goals or challenges they're facing, tailor your response to address those specific needs.
    """

# Chat endpoint
@app.post("/api/chat", response_description="Process a chat message")
async def process_chat_message(request: ChatRequest, client_request: Request):
    """Process a chat message and return a response"""
    try:
        # Add user_id from auth token if available
        token = get_token_from_header(client_request)
        if token:
            request.user_id = token
        
        # Don't cache chat responses, as they should be dynamic
        response = await call_llm_api(request.dict(), "chat")
        
        # Store conversation in Firebase if needed for continuity
        # This is optional but helpful for maintaining conversation context
        user_id = request.user_id
        db.collection('conversations').document(user_id).set({
            'last_message': request.message,
            'last_response': response.get('output', ''),
            'timestamp': time.time()
        }, merge=True)
        
        return response
    except Exception as e:
        logger.error(f"Chat error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# API endpoints
@app.post("/api/routine", response_description="Generate a personalized routine plan")
async def generate_routine(request: RoutineRequest, client_request: Request):
    """Generate a routine plan based on user parameters"""
    # Generate cache key from request data
    cache_key = generate_cache_key(request.dict())
    
    # Check cache first
    cached_response = await check_cache(cache_key)
    if cached_response:
        return cached_response
    
    # Make request to LLM API
    try:
        response = await call_llm_api(request.dict(), "routine")
        
        # Save to cache
        await save_to_cache(cache_key, response)
        
        # Store the original request in Firebase for future follow-ups
        user_id = get_token_from_header(client_request) or "anonymous"
        db.collection('user_requests').document(user_id).set({
            'last_routine_request': request.dict(),
            'timestamp': time.time()
        }, merge=True)
        
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/nutrition", response_description="Generate a personalized nutrition plan")
async def generate_nutrition_plan(request: NutritionRequest, client_request: Request):
    """Generate a nutrition plan based on user parameters"""
    # Generate cache key from request data
    cache_key = generate_cache_key(request.dict())
    
    # Check cache first
    cached_response = await check_cache(cache_key)
    if cached_response:
        return cached_response
    
    # Get peak mode and challenges from user's last routine request if available
    user_id = get_token_from_header(client_request) or "anonymous"
    user_doc = db.collection('user_requests').document(user_id).get()
    
    request_data = request.dict()
    
    # Add peak mode and challenges from routine request if available
    if user_doc.exists:
        user_data = user_doc.to_dict()
        if 'last_routine_request' in user_data:
            routine_data = user_data['last_routine_request']
            request_data['peak_mode'] = routine_data.get('peak_mode')
            request_data['challenges'] = routine_data.get('challenges', [])
    
    # Make request to LLM API
    try:
        response = await call_llm_api(request_data, "nutrition")
        
        # Save to cache
        await save_to_cache(cache_key, response)
        
        # Store the original request in Firebase
        db.collection('user_requests').document(user_id).set({
            'last_nutrition_request': request.dict(),
            'timestamp': time.time()
        }, merge=True)
        
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/follow_up", response_description="Handle follow-up questions about a plan")
async def handle_follow_up(request: FollowUpRequest, client_request: Request):
    """Handle follow-up questions by referencing previous context"""
    # Make request to LLM API
    try:
        response = await call_llm_api(request.dict(), "follow_up")
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health", response_description="Check API health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "health-coach-api"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)