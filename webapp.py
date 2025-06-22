import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import tflite_runtime.interpreter as tflite  # Light and compatible

# Set page config
st.set_page_config(page_title="🧠 Digit Generator", layout="wide")

st.title("🧠 Handwritten Digit Generator")
st.markdown("This app uses a TFLite model to generate handwritten digits using a Variational Autoencoder.")

# Load the TFLite model
@st.cache_resource
def load_model():
    interpreter = tflite.Interpreter(model_path="vae_decoder.tflite")
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    return interpreter, input_details, output_details

interpreter, input_details, output_details = load_model()

# Create layout
st.subheader("🖼️ Generated Digit Variations")

cols = st.columns(5)

for i in range(5):
    with cols[i]:
        # Sample random latent vector
        z = np.random.normal(size=(1, 2)).astype(np.float32)

        # Set input tensor
        interpreter.set_tensor(input_details[0]['index'], z)
        interpreter.invoke()

        # Get output
        output = interpreter.get_tensor(output_details[0]['index'])
        image = output[0].reshape(8, 8)

        # Show image
        plt.imshow(image, cmap="gray")
        plt.axis("off")
        st.pyplot(plt)
