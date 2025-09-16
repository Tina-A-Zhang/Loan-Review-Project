import pickle
from tensorflow.keras.models import load_model

# Load your h5 model
model = load_model("trained_model.h5")

# Save as pickle
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)
print("✅ Model converted to model.pkl")
