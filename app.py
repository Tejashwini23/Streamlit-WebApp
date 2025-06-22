import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import tflite_runtime.interpreter as tflite  # Lightweight TFLite runtime

# Streamlit config
st.set_page_config(page_title="🧠 Digit Generator", layout="wide")
st.title("🧠 Handwritten Digit Generator")
st.markdown("This app uses a TFLite model to generate handwritten digits using a Variational Autoencoder (VAE).")

# Load the TFLite model (cached to avoid reloading)
@st.cache_resource
def load_model():
    interpreter = tflite.Interpreter(model_path="vae_decoder.tflite")
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    return interpreter, input_details, output_details

interpreter, input_details, output_details = load_model()

# UI section
st.subheader("🖼️ Generated Digit Variations")

# Display 5 generated digits side by side
cols = st.columns(5)

for i in range(5):
    with cols[i]:
        # Generate a random 2D latent vector
        z = np.random.normal(size=(1, 2)).astype(np.float32)

        # Run inference
        interpreter.set_tensor(input_details[0]['index'], z)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])

        # Reshape and plot the image
        image = output[0].reshape(8, 8)
        fig, ax = plt.subplots()
        ax.imshow(image, cmap="gray")
        ax.axis("off")
        st.pyplot(fig)
