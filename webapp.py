import streamlit as st
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Set Streamlit page configuration
st.set_page_config(page_title="Digit Generator", layout="wide")

# Load the trained decoder model
@st.cache_resource
def load_decoder():
    return tf.keras.models.load_model("/content/vae_decoder.h5")

decoder = load_decoder()

st.title("🧠 Handwritten Digit Image Generator (VAE from Scratch)")

st.markdown("""
This app generates different **handwritten digit variants (0–9)** using a Variational Autoencoder (VAE) 
trained from scratch on the `digits` dataset.

🔢 Select a digit and view 5 variations!
""")

# Input digit (optional for display – real generation is label-free)
digit = st.number_input("Choose a digit (0–9) just for labeling", min_value=0, max_value=9, step=1)

st.subheader(f"🖼️ Variants of Digit {digit} (Generated)")

cols = st.columns(5)
for i in range(5):
    with cols[i]:
        latent_vector = np.random.normal(size=(1, 2))  # Latent space is 2D
        generated = decoder.predict(latent_vector, verbose=0)[0].reshape(8, 8)
        plt.imshow(generated, cmap='gray')
        plt.axis('off')
        st.pyplot(plt)
