import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import gradio as gr
import matplotlib.pyplot as plt
import time

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
    "inverse_wc_ratio": [1/x for x in w_c_ratio],  # Adding inverse to help model learn trend
    "curing_days": curing_days,
    "fly_ash_content": fly_ash_content,
    "compressive_strength": compressive_strength
}

df = pd.DataFrame(data)

# 🔹 Define input (X) and output (y)
X = df[['w/c_ratio', 'inverse_wc_ratio', 'curing_days', 'fly_ash_content']]
y = df['compressive_strength']

# 🔹 Standardize input features for better learning
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 🔹 Split data into training (80%) and testing (20%)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 🔹 Improved ANN model with L2 Regularization
model = Sequential([
    Dense(128, activation='relu', kernel_regularizer=l2(0.01), input_shape=(X_train.shape[1],)),
    Dense(64, activation='relu', kernel_regularizer=l2(0.01)),
    Dense(32, activation='relu', kernel_regularizer=l2(0.01)),
    Dense(1, activation='linear')  # Output layer for regression
])

# 🔹 Compile the model with Huber Loss
model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
              loss=tf.keras.losses.Huber(),
              metrics=['mae'])

# 🔹 Train the model with Early Stopping to prevent overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=True)
model.fit(X_train, y_train, epochs=1500, batch_size=8, validation_data=(X_test, y_test), verbose=0, callbacks=[early_stopping])

# 🔹 Save trained model
model.save("ann_model.h5")

# 🔹 Load model
model = keras.models.load_model("ann_model.h5", custom_objects={"Huber": tf.keras.losses.Huber()})

# 🔹 Function to predict compressive strength
def predict_strength(wc_ratio, curing_days, fly_ash):
    input_data = np.array([[wc_ratio, 1/wc_ratio, curing_days, fly_ash]])
    input_scaled = scaler.transform(input_data)  # Apply same scaling
    prediction = model.predict(input_scaled)[0][0]

    # Generate corrected trend graph
    curing_days_range = np.linspace(7, 90, 15)
    predicted_values = [model.predict(scaler.transform([[wc_ratio, 1/wc_ratio, d, fly_ash]]))[0][0] for d in curing_days_range]

    plt.figure(figsize=(7, 5))
    plt.plot(curing_days_range, predicted_values, marker='o', linestyle='-', color='blue', label="Predicted Strength")
    plt.xlabel("Curing Days", fontsize=13)
    plt.ylabel("Predicted Strength (MPa)", fontsize=13)
    plt.title("Strength Prediction Trend", fontsize=15, fontweight='bold', color='blue')
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend()
    plt.savefig("prediction_plot.png")
    plt.close()
    
    return prediction, "prediction_plot.png"

# 🔹 Gradio UI for user interaction
with gr.Blocks(css="body {background-color: #f8f9fa;}") as demo:
    gr.Markdown("""
    # 🏛️ **Compressive Strength Predictor**
    
    Enter details below to predict compressive strength
    """)
    
    with gr.Row():
        wc_input = gr.Number(label="Enter w/c ratio")
        days_input = gr.Number(label="Enter curing days")
    
    fly_ash_dropdown = gr.Dropdown([22, 33], label="Fly Ash % in PPC", value=22)
    
    predict_button = gr.Button("Predict", variant="primary")
    
    output_strength = gr.Number(label="Predicted Strength (MPa)")
    output_graph = gr.Image()
    
    predict_button.click(fn=predict_strength, inputs=[wc_input, days_input, fly_ash_dropdown], outputs=[output_strength, output_graph])
    
    gr.Markdown("""
    **📊 Strength Trend Graph:**
    - The graph shows predicted strength variation over curing periods.
    - This helps in analyzing long-term strength gain trends.
    """)

demo.launch(share=True)
