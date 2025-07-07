import tkinter as tk
from tkinter import messagebox
import numpy as np
import joblib

# Load trained model only (no scaler needed)
model = joblib.load('rock_mine_model.sav')

# GUI setup
window = tk.Tk()
window.title("Rock vs Mine Predictor")
window.geometry("500x700")
window.configure(bg="#f0f0f0")

tk.Label(window, text="Enter 60 values (comma-separated)", font=("Arial", 14, "bold"), bg="#f0f0f0").pack(pady=10)

text_input = tk.Text(window, height=15, width=60, font=("Courier", 10))
text_input.pack()

def predict():
    try:
        # Read input from text box
        input_data = text_input.get("1.0", tk.END).strip()
        values = [float(i) for i in input_data.split(',')]
        
        if len(values) != 60:
            messagebox.showerror("Error", "Please enter exactly 60 numerical values.")
            return

        input_np = np.asarray(values).reshape(1, -1)

        prediction = model.predict(input_np)[0]

        result = "🔵 The object is **ROCK**" if prediction == 'R' else "🟡 The object is **MINE**"
        messagebox.showinfo("Prediction Result", result)

    except ValueError:
        messagebox.showerror("Error", "Invalid input! Please enter numeric values only.")

tk.Button(window, text="Predict", font=("Arial", 12), command=predict, bg="#4CAF50", fg="white").pack(pady=20)

window.mainloop()
