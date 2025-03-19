# locustfile.py
from locust import HttpUser, task, between
import json
import random

class HealthCoachUser(HttpUser):
    wait_time = between(1, 5)  # Wait between 1-5 seconds between tasks
    
    def on_start(self):
        # Add auth if needed
        self.client.headers = {"Authorization": "Bearer test-api-key"}
    
    @task(1)
    def health_check(self):
        self.client.get("/health")
    
    @task(3)
    def get_routine_plan(self):
        # Create routine request with slight variations for less caching
        peak_modes = ["Physique", "Productivity", "Cognition", "Longevity"]
        chronotypes = ["Early Bird", "Second Bird", "Third Bird", "Night Owl"]
        commutes = ["Low Commute", "Medium Commute", "High Commute with long driving periods"]
        travel = ["Less than 4 times per year", "4-8 times per year", "Monthly", "Weekly"]
        challenges = [
            "Low nutrition IQ",
            "prolonged sedentary periods",
            "midday sluggishness",
            "routine is hard to maintain",
            "high stress levels",
            "poor sleep quality"
        ]
        
        # Generate randomized but realistic request
        request_data = {
            "peak_mode": random.choice(peak_modes),
            "sleep_chronotype": random.choice(chronotypes),
            "nature_of_commutes": random.choice(commutes),
            "nature_of_traveling": random.choice(travel),
            "easiness_of_regimen": random.randint(5, 9),
            "observation_level": random.randint(5, 9),
            "challenges": random.sample(challenges, k=random.randint(2, 4))
        }
        
        # Send request
        response = self.client.post("/api/routine", json=request_data)
        
        # Log failures
        if response.status_code != 200:
            print(f"Routine request failed: {response.status_code}, {response.text}")
    
    @task(2)
    def get_nutrition_plan(self):
        # Create randomized nutrition request
        protein_sources = ["Mixed", "Plant-based", "Animal-based", "Pescatarian"]
        carb_tolerances = ["Higher carb diet", "Balanced carb diet", "Lower carb diet", "Ketogenic"]
        fat_preferences = ["Higher fat", "Moderate fat", "Lower fat", "None"]
        cooking_abilities = ["Professional", "Good home cook", "Basic cooking skills", "No ability. Less than 6hrs/wk."]
        supplements = [
            "Protein Powder",
            "Creatine",
            "Fish Oil",
            "Zinc",
            "Vitamin D",
            "Magnesium",
            "Probiotic"
        ]
        
        # Generate request data
        request_data = {
            "current_body_composition": {
                "weight": random.randint(150, 300),
                "body_fat_percentage": round(random.uniform(15, 40), 1),
                "lean_body_mass": round(random.uniform(120, 200), 1)
            },
            "target_goal": random.choice(["Weight Loss", "Muscle Gain", "Recomposition", "Maintenance"]),
            "activity_level": random.choice(["Sedentary", "Lightly Active", "Moderately Active", "Very Active"]),
            "daily_step_count": random.randint(4000, 12000),
            "resting_metabolic_rate": random.randint(1800, 2500),
            "macronutrient_preferences": random.choice(["Higher Protein", "Higher Carb", "Higher Fat", "Balanced"]),
            "intermittent_fasting": random.choice([True, False]),
            "num_meals_per_day": random.randint(3, 6),
            "eating_window": random.choice(["7am to 7pm", "9am to 8pm", "12pm to 8pm"]),
            "protein_source_preference": random.choice(protein_sources),
            "carb_tolerance": random.choice(carb_tolerances),
            "fat_preference": random.choice(fat_preferences),
            "food_sensitivities": random.sample(["Dairy", "Gluten", "Eggs", "Soy", "Nuts"], k=random.randint(0, 2)),
            "cultural_dietary_category": random.choice(["None", "Mediterranean", "Asian", "Latin"]),
            "cooking_ability": random.choice(cooking_abilities),
            "meal_prep_frequency": random.choice(["Daily", "2-3 times per week", "Once per week"]),
            "eating_out_frequency": random.choice(["Never", "1-2 times per week", "3-5 times per week"]),
            "kitchen_access": random.choice(["Full kitchen", "Limited kitchen", "Minimal kitchen"]),
            "existing_supplements": random.sample(supplements, k=random.randint(1, 4)),
            "willingness_to_supplement": random.choice([True, False]),
            "water_intake_target": f"{random.randint(2, 5)}.{random.randint(0, 9)}L",
            "caffeine_consumption": random.choice(["None", "Low", "Moderate", "High"]),
            "electrolyte_needs": random.choice(["Standard", "High (heavy sweating)", "Indulgent salt intake, Minimum sweating"])
        }
        
        # Send request
        response = self.client.post("/api/nutrition", json=request_data)
        
        # Log failures
        if response.status_code != 200:
            print(f"Nutrition request failed: {response.status_code}, {response.text}")

# Running the load test:
# locust -f locustfile.py --host=http://localhost:8000
# Then open http://localhost:8089 in your browser