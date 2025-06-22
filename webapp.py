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
        # Use the new GAN generator model file
        model = tf.keras.models.load_model("gan_generator.h5")
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

    # Standard GANs use a higher-dimensional latent space (e.g., 100 dimensions)
    # This gives them more creative freedom to generate diverse and sharp images.
    LATENT_DIM = 100

    # --- Setup columns for controls and images ---
    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Controls")
        if st.button("Generate New Random Start/End Digits"):
            # Store two random latent vectors in session state
            st.session_state.z_start = np.random.normal(size=(1, LATENT_DIM))
            st.session_state.z_end = np.random.normal(size=(1, LATENT_DIM))

        # Initialize the latent vectors if they don't exist
        if "z_start" not in st.session_state:
            st.session_state.z_start = np.random.normal(size=(1, LATENT_DIM))
            st.session_state.z_end = np.random.normal(size=(1, LATENT_DIM))

        # The main slider for interpolation
        morph_level = st.slider(
            "Morphing Level (from Start to End)",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.01
        )

    # --- Generation and Display Logic ---
    with col2:
        st.subheader("Generated Image")

        # 1. Interpolate between the start and end vectors based on the slider
        z_interpolated = (
            st.session_state.z_start * (1 - morph_level) +
            st.session_state.z_end * morph_level
        ).astype(np.float32)

        # 2. Generate the image from the interpolated latent vector
        generated_image_raw = generator.predict(z_interpolated)

        # 3. Post-process the image for display
        # The model outputs values between -1 and 1. We shift them to 0-1.
        generated_image = (generated_image_raw[0] + 1) / 2.0
        # Reshape to 28x28 since this GAN was trained on 28x28 MNIST
        generated_image = generated_image.reshape(28, 28)

        # 4. Display the sharp, high-contrast image
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.imshow(generated_image, cmap='binary', interpolation='nearest')
        ax.axis("off") # Hide the x and y axes
        st.pyplot(fig)


    # --- Display the start and end images for context ---
    st.subheader("Start & End Points")
    col_start, col_end = st.columns(2)
    with col_start:
        start_image = (generator.predict(st.session_state.z_start)[0] + 1) / 2.0
        fig, ax = plt.subplots()
        ax.imshow(start_image.reshape(28, 28), cmap='binary')
        ax.set_title("Start Image")
        ax.axis("off")
        st.pyplot(fig)

    with col_end:
        end_image = (generator.predict(st.session_state.z_end)[0] + 1) / 2.0
        fig, ax = plt.subplots()
        ax.imshow(end_image.reshape(28, 28), cmap='binary')
        ax.set_title("End Image")
        ax.axis("off")
        st.pyplot(fig)
