# main.py with SPiR Chatbot Workflow v2
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
from typing import Dict, Any, Optional, List, Literal

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
    # allow_origins=["https://bbxsh-93b83.web.app"],  # Only allow your frontend domain
    allow_origins=["https://bbxsh-93b83.web.app", "http://localhost:51295", "http://127.0.0.1:8000","https://localhost:3000/", "https://www.spir.health/","https://staging.spir.health/"],
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

# Updated Models for SPiR Workflow v2
class RoutineRequest(BaseModel):
    main_goal: Literal[
        "Sharper Focus/Brainpower",
        "More Energy", 
        "Physique Change",
        "Stress Reduction/Recovery"
    ]
    energy_rhythm: Literal[
        "Morning Alertness",
        "Evening Alertness",
        "Flexible"
    ]
    wake_up_time: Literal[
        "Early Bird (4/5am)",
        "Third Bird (6/7am)",
        "Night Owl (8/9am)"
    ]
    biggest_challenges: List[Literal[
        "Sticking to a routine",
        "Low energy",
        "Poor Sleep",
        "High stress",
        "Inconsistent Diet",
        "Prolonged sitting",
        "Exercise motivation",
        "Screen time",
        "Obligations",
        "Other"
    ]]
    typical_day_structure: Literal[
        "WFH",
        "Hybrid",
        "On-site",
        "Travel"
    ]
    exercise_preference: Literal[
        "Morning",
        "Midday",
        "Evening",
        "None"
    ]
    exercise_context: Literal[
        "Gym/Studio (Weights + Cardio)",
        "Gym/Studio (Weights)",
        "Gym/Studio (Cardio)",
        "Home (Weights + Cardio)",
        "Home (Weights)",
        "Home (Cardio)",
        "Various",
        "None"
    ]
    sleep_quality: Dict[str, Any]  # Contains trouble_falling_asleep, wake_during_night, wake_tired, wind_down_routine, medical_conditions
    additional_notes: Optional[str] = ""

class NutritionRequest(BaseModel):
    main_goal: Literal[
        "Weight Loss",
        "Muscle Gain",
        "Maintenance",
        "Overall Health"
    ]
    dietary_preference: Literal[
        "Balanced",
        "High Protein / Low Carb",
        "Plant-Based",
        "Mediterranean",
        "Keto"
    ]
    meal_preference: List[Literal[
        "Quick & Easy / Meal Prep Friendly / Budget Friendly",
        "Variety of Flavors",
        "Low Sugar",
        "High Fiber",
        "Low Sodium",
        "Family Friendly"
    ]]
    daily_schedule: Literal[
        "Early Bird",
        "Night Owl",
        "Standard",
        "Shift Worker"
    ]
    dietary_restrictions: List[Literal[
        "Gluten-Free",
        "Dairy-Free",
        "Vegetarian",
        "Vegan",
        "Nut-Free",
        "Shellfish-Free",
        "Soy-Free",
        "Egg-Free",
        "Allergies"
    ]]
    preferred_meal_pattern: Literal[
        "3 meals (no snacks)",
        "3 meals + 1 snack",
        "3 meals + 2 snacks",
        "6 small meals"
    ]
    medical_considerations: Optional[List[str]] = []
    additional_notes: Optional[str] = ""

# Also add a new model for follow-up questions (optional for implementation)
class FollowUpRequest(BaseModel):
    user_id: str
    original_request_type: str  # "routine" or "nutrition"
    original_request_data: Dict[str, Any]
    follow_up_question: str

class ChatRequest(BaseModel):
    message: str
    user_id: Optional[str] = "anonymous"
    conversation_history: Optional[List[Dict[str, Any]]] = []

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
    
    # Updated system prompt for SPiR v2
    system_prompt = """You are a master health coach playing the role of SPiR, highly trained in nutrition science, chronobiology, and cognitive behavioral science. 
    You provide actionable suggestions with respect to the circadian energy zones of the day. Your aim is to give users actionable insights on how to properly structure the most intuitive routine for their day.
    When finished, the user should know what habits, behavior or advice to adhere to as well as know the purposes in relation to their goal.
    Always structure your responses by the circadian energy zones and connect recommendations to the user's specific goals and challenges."""
    
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

# Updated prompt construction functions for SPiR v2
def construct_routine_prompt(data: dict) -> str:
    """Construct a prompt for routine planning based on SPiR v2 workflow"""
    
    sleep_issues = []
    if data['sleep_quality'].get('trouble_falling_asleep'):
        sleep_issues.append("trouble falling asleep")
    if data['sleep_quality'].get('wake_during_night'):
        sleep_issues.append("waking up during the night")
    if data['sleep_quality'].get('wake_tired'):
        sleep_issues.append("waking up tired")
    
    return f"""
Create a personalized daily routine for a user with the following profile:

PROFILE:
Main Goal: {data['main_goal']}
Energy Rhythm: {data['energy_rhythm']}
Wake Up Time: {data['wake_up_time']}
Biggest Challenges: {', '.join(data['biggest_challenges'])}
Typical Day Structure: {data['typical_day_structure']}
Exercise Preference: {data['exercise_preference']}
Exercise Context: {data['exercise_context']}
Sleep Quality Issues: {', '.join(sleep_issues) if sleep_issues else 'None'}
Wind-down Routine: {data['sleep_quality'].get('wind_down_routine', 'No')}
Medical Conditions: {data['sleep_quality'].get('medical_conditions', 'None')}
Additional Notes: {data.get('additional_notes', 'None')}

HIGH PERFORMANCE GUIDELINES TO INCORPORATE:
1. Routine Management: Proper goal setting & tracking, regular decompression, time-blocked agenda
2. Sleep Hygiene: 7hrs/night (42hrs/week), sleep HR <60bpm, intentional wind-down routine
3. Exercise Protocols: 8,000-10,000 steps/day, 6-8 training hours/week, daily stretching, resting HR <70bpm, VO2 Max >50
4. Nutrition Strategy: Fasted 3-5 hours after wake and before sleep, minimum 2L water/day, circadian-friendly diet, balanced macros

Structure your response by CIRCADIAN ENERGY ZONES:

🌅 EARLY MORNING (Wake to 2 hours after)
[Explain this energy phase and its significance for their goal]
• [Specific recommendation with timing]
• [Specific recommendation with reasoning]
• [Connect to their challenges/goals]

☀️ MORNING PEAK (2-5 hours after wake)
[Explain peak performance window for their chronotype]
• [Deep work recommendations]
• [Energy optimization strategies]
• [Goal-specific activities]

🌤️ MIDDAY TRANSITION (5-8 hours after wake)
[Explain natural energy dip and recovery strategies]
• [Lunch timing and composition]
• [Movement/recovery activities]
• [Stress management techniques]

⚡ AFTERNOON SURGE (8-12 hours after wake)
[Explain second wind utilization]
• [Task prioritization for this energy level]
• [Exercise timing if applicable]
• [Productivity strategies]

🌆 EVENING WIND-DOWN (12-15 hours after wake)
[Explain transition to rest phase]
• [Dinner timing and composition]
• [Screen time management]
• [Relaxation protocols]

🌙 PRE-SLEEP ROUTINE (Last 2 hours before bed)
[Explain sleep preparation for their specific issues]
• [Wind-down activities]
• [Sleep hygiene practices]
• [Environment optimization]

End with a brief summary connecting this routine to their main goal of {data['main_goal']} and how it addresses their specific challenges.
"""

def construct_nutrition_prompt(data: dict) -> str:
    """Construct a prompt for nutrition planning based on SPiR v2 workflow"""
    
    return f"""
Create a personalized nutrition plan for a user with the following profile:

PROFILE:
Main Goal: {data['main_goal']}
Dietary Preference: {data['dietary_preference']}
Meal Preferences: {', '.join(data['meal_preference'])}
Daily Schedule: {data['daily_schedule']}
Dietary Restrictions: {', '.join(data['dietary_restrictions']) if data['dietary_restrictions'] else 'None'}
Preferred Meal Pattern: {data['preferred_meal_pattern']}
Medical Considerations: {', '.join(data.get('medical_considerations', [])) if data.get('medical_considerations') else 'None'}
Additional Notes: {data.get('additional_notes', 'None')}

HIGH PERFORMANCE NUTRITION GUIDELINES:
1. Fasting Windows: 3-5 hours after wake and before sleep
2. Hydration: Minimum 2L water/day
3. Circadian-Friendly: Align eating with natural energy rhythms
4. Balanced Macros: Appropriate for their goal
5. Gut Health: Include fermented foods and fiber

Structure your response by CIRCADIAN EATING ZONES:

🌅 BREAK-FAST WINDOW (Based on their schedule type: {data['daily_schedule']})
[Explain optimal first meal timing for their circadian rhythm]
• Meal Option 1: [Specific meal with macros]
• Meal Option 2: [Alternative with explanation]
• Hydration Protocol: [Morning hydration strategy]
• Why this timing supports {data['main_goal']}

☀️ MIDDAY FUEL (Peak Energy Hours)
[Explain lunch timing for sustained energy]
• Meal Option 1: [Detailed meal plan]
• Meal Option 2: [Alternative option]
• Pre/Post meal movement suggestions
• Energy optimization tips

⚡ AFTERNOON NUTRITION (If applicable based on {data['preferred_meal_pattern']})
[Explain snack/meal timing and purpose]
• Snack/Meal options
• Portion guidance
• Timing relative to exercise

🌆 EVENING MEAL (Final eating window)
[Explain dinner timing for recovery and sleep]
• Meal Option 1: [Detailed with cooking method]
• Meal Option 2: [Quick alternative]
• Nutrient timing for overnight recovery
• Why early dinner supports their goals

💧 HYDRATION & SUPPLEMENTATION SCHEDULE
• Morning: [Specific recommendations]
• Midday: [Hydration targets]
• Evening: [Cut-off times]
• Supplement timing (if applicable)

📋 WEEKLY MEAL PREP STRATEGY
[Provide specific prep tips for their preferences: {', '.join(data['meal_preference'])}]
• Sunday prep list
• Make-ahead options
• Storage tips
• Quick assembly meals

End with specific tips for maintaining this nutrition plan with their {data['typical_day_structure'] if 'typical_day_structure' in data else data['daily_schedule']} schedule and how it directly supports their goal of {data['main_goal']}.
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
        Main Goal: {original_data.get('main_goal')}
        Energy Rhythm: {original_data.get('energy_rhythm')}
        Challenges: {', '.join(original_data.get('biggest_challenges', []))}
        Day Structure: {original_data.get('typical_day_structure')}
        """
    else:  # nutrition
        context = f"""
        The client previously received a nutrition plan with these parameters:
        Main Goal: {original_data.get('main_goal')}
        Dietary Preference: {original_data.get('dietary_preference')}
        Meal Pattern: {original_data.get('preferred_meal_pattern')}
        Schedule: {original_data.get('daily_schedule')}
        """
    
    return f"""
    {context}
    
    Now they're asking: "{follow_up_question}"
    
    Respond in the same format organized by CIRCADIAN ENERGY ZONES or CIRCADIAN EATING ZONES as appropriate.
    Ensure your response directly addresses their question while maintaining continuity with their previous plan.
    Keep the warm, coaching tone and provide specific, actionable recommendations that align with the HIGH PERFORMANCE GUIDELINES.
    """

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
            The user previously created a routine plan with main goal: {routine.get('main_goal')}
            and identified challenges: {', '.join(routine.get('biggest_challenges', []))}
            """
        
        if 'last_nutrition_request' in user_data:
            nutrition = user_data['last_nutrition_request']
            user_context += f"""
            The user previously created a nutrition plan with main goal: {nutrition.get('main_goal')}
            and dietary preference: {nutrition.get('dietary_preference')}
            """
    
    return f"""
    You are SPiR, a master health coach specializing in circadian-optimized routines and nutrition.
    
    {user_context}
    
    The user has sent the following message: "{message}"
    
    Respond as SPiR in a warm, conversational tone. Be helpful and supportive while providing actionable advice based on circadian science and the HIGH PERFORMANCE GUIDELINES.
    
    If they're asking about creating a detailed routine or nutrition plan, suggest they use the dedicated forms in the app for the best personalized experience aligned with circadian rhythms.
    
    If they have a specific health or nutrition question, provide thoughtful guidance based on chronobiology and your expertise, while being careful not to make medical claims.
    
    Always try to connect your advice to circadian energy zones when relevant.
    """

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
    
    # Get user's routine data if available to align nutrition with their schedule
    user_id = get_token_from_header(client_request) or "anonymous"
    user_doc = db.collection('user_requests').document(user_id).get()
    
    request_data = request.dict()
    
    # Add routine data to nutrition request for better alignment
    if user_doc.exists:
        user_data = user_doc.to_dict()
        if 'last_routine_request' in user_data:
            routine_data = user_data['last_routine_request']
            request_data['typical_day_structure'] = routine_data.get('typical_day_structure')
            request_data['wake_up_time'] = routine_data.get('wake_up_time')
            request_data['exercise_preference'] = routine_data.get('exercise_preference')
    
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

@app.get("/health", response_description="Check API health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "health-coach-api"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
