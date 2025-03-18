import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests
from fastapi import FastAPI
from pydantic import BaseModel
from starlette.responses import FileResponse

# Hugging Face API Endpoint
HF_API_URL = "YOUR_HUGGINGFACE_MODEL_ENDPOINT"

app = FastAPI()

class PredictionInput(BaseModel):
    w_c_ratio: float
    curing_days: int
    fly_ash_content: int

# 🔹 Manually inserting dataset (Filtered: Removed 0.48 w/c ratio)
w_c_ratio = [0.36, 0.40, 0.44, 0.52] * 4
curing_days = [7] * 8 + [28] * 8
fly_ash_content = [22] * 4 + [33] * 4 + [22] * 4 + [33] * 4
compressive_strength = [
    22.82, 20.17, 17.46, 12.79,  
    18.68, 16.78, 15.15, 10.20,  
    30.36, 28.78, 27.01, 22.50,  
    29.79, 26.67, 24.62, 15.47   
]

data = {
    "w/c_ratio": w_c_ratio,
    "curing_days": curing_days,
    "fly_ash_content": fly_ash_content,
    "compressive_strength": compressive_strength
}

df = pd.DataFrame(data)

# 🔹 Define input (X) and output (y)
X = df[['w/c_ratio', 'curing_days', 'fly_ash_content']]
y = df['compressive_strength']

def query_huggingface(payload):
    response = requests.post(HF_API_URL, json=payload)
    return response.json()

@app.post("/predict")
def predict_strength(data: PredictionInput):
    input_data = {
        "w/c_ratio": data.w_c_ratio,
        "curing_days": data.curing_days,
        "fly_ash_content": data.fly_ash_content
    }
    prediction = query_huggingface(input_data)
    predicted_strength = prediction.get("predicted_strength", 0)

    # Generate corrected trend graph
    curing_days_range = np.linspace(7, 90, 15)
    predicted_values = [
        query_huggingface({"w/c_ratio": data.w_c_ratio, "curing_days": d, "fly_ash_content": data.fly_ash_content}).get("predicted_strength", 0)
        for d in curing_days_range
    ]

    plt.figure(figsize=(7, 5))
    plt.plot(curing_days_range, predicted_values, marker='o', linestyle='-', color='blue', label="Predicted Strength")
    plt.xlabel("Curing Days", fontsize=13)
    plt.ylabel("Predicted Strength (MPa)", fontsize=13)
    plt.title("Strength Prediction Trend", fontsize=15, fontweight='bold', color='blue')
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend()
    plt.savefig("prediction_plot.png")
    plt.close()

    return {"predicted_strength": predicted_strength, "graph": "/graph"}

@app.get("/graph")
def get_graph():
    return FileResponse("prediction_plot.png", media_type="image/png")
