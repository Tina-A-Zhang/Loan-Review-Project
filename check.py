import tensorflow as tf
import numpy as np


model = tf.keras.models.load_model("your_model.h5")
print("✅ Model loaded successfully!")


model.summary()


dummy_input = np.random.random((1, model.input_shape[1]))
pred = model.predict(dummy_input)
print("Prediction output:", pred)


model.save("your_model_full.h5")
print("✅ Model saved as your_model_full.h5")
