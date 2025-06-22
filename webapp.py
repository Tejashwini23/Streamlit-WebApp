import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# Set Streamlit page config
st.set_page_config(page_title="🧠 Digit Generator (Keras)", layout="wide")
st.title("🧠 Handwritten Digit Generator")
st.markdown("This app uses a trained Keras `.h5` decoder model to generate handwritten digits.")

# Load the model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("vae_decoder.h5")

model = load_model()

# Generate and display digits
st.subheader("🖼️ Generated Digit Samples")
cols = st.columns(5)

for i in range(5):
    with cols[i]:
        # Sample a random latent vector (2D)
        z = np.random.normal(size=(1, 2)).astype(np.float32)

        # Generate output using the decoder
        output = model.predict(z)

        # Reshape and display the image (8x8 assumed)
        image = output[0].reshape(8, 8)
        fig, ax = plt.subplots()
        ax.imshow(image, cmap='gray')
        ax.axis("off")
        st.pyplot(fig)
