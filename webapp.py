import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# --- Page Setup ---
st.set_page_config(page_title="🧠 GAN Digit Generator", layout="wide")
st.title("🧠 Sharp Handwritten Digit Generator (GAN)")
st.markdown("This app uses a pre-trained **GAN (Generative Adversarial Network)** to generate sharp, clear 28x28 handwritten digits. GANs produce much higher quality images than VAEs.")

# --- Load the GAN Generator Model ---
@st.cache_resource
def load_model():
    """Loads the pre-trained Keras GAN Generator model."""
    try:
        model = tf.keras.models.load_model("vae_devoder.h5", compile=False) # Added compile=False for robustness
        return model
    except FileNotFoundError:
        st.error("Model file 'gan_generator.h5' not found. Please download it and place it in the same directory.")
        return None
    except Exception as e:
        st.error(f"An error occurred while loading the model: {e}")
        return None

generator = load_model()

# --- Main App Logic ---
if generator:
    st.header("🎨 Interactive Digit Explorer")
    st.markdown("Generate two random digits, then use the slider to smoothly **morph** between them!")

    LATENT_DIM = 100

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Controls")
        if st.button("Generate New Random Start/End Digits"):
            st.session_state.z_start = np.random.normal(size=(1, LATENT_DIM)).astype(np.float32)
            st.session_state.z_end = np.random.normal(size=(1, LATENT_DIM)).astype(np.float32)

        if "z_start" not in st.session_state:
            st.session_state.z_start = np.random.normal(size=(1, LATENT_DIM)).astype(np.float32)
            st.session_state.z_end = np.random.normal(size=(1, LATENT_DIM)).astype(np.float32)

        morph_level = st.slider(
            "Morphing Level (from Start to End)", 0.0, 1.0, 0.5, 0.01
        )

    with col2:
        st.subheader("Generated Image")

        z_interpolated = (
            st.session_state.z_start * (1 - morph_level) +
            st.session_state.z_end * morph_level
        )
        
        # Explicitly convert the numpy array to a TensorFlow tensor before prediction
        z_tensor = tf.convert_to_tensor(z_interpolated) # <-- FIX HERE

        generated_image_raw = generator.predict(z_tensor) # <-- FIX HERE

        generated_image = (generated_image_raw[0] + 1) / 2.0
        generated_image = generated_image.reshape(28, 28)

        fig, ax = plt.subplots(figsize=(5, 5))
        ax.imshow(generated_image, cmap='binary', interpolation='nearest')
        ax.axis("off")
        st.pyplot(fig)

    st.subheader("Start & End Points")
    col_start, col_end = st.columns(2)
    with col_start:
        # Also apply the fix to the other predict calls
        start_tensor = tf.convert_to_tensor(st.session_state.z_start) # <-- FIX HERE
        start_image = (generator.predict(start_tensor)[0] + 1) / 2.0 # <-- FIX HERE
        
        fig, ax = plt.subplots()
        ax.imshow(start_image.reshape(28, 28), cmap='binary')
        ax.set_title("Start Image")
        ax.axis("off")
        st.pyplot(fig)

    with col_end:
        # Also apply the fix to the other predict calls
        end_tensor = tf.convert_to_tensor(st.session_state.z_end) # <-- FIX HERE
        end_image = (generator.predict(end_tensor)[0] + 1) / 2.0 # <-- FIX HERE
        
        fig, ax = plt.subplots()
        ax.imshow(end_image.reshape(28, 28), cmap='binary')
        ax.set_title("End Image")
        ax.axis("off")
        st.pyplot(fig)
