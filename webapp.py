import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# --- Page Setup ---
st.set_page_config(page_title="🧠 Digit Generator", layout="wide")
st.title("🧠 Handwritten Digit Generator")
st.markdown("This app uses a Keras VAE decoder to generate handwritten digits. Choose a digit and see the variations!")

# --- Load the Model (Cached for Performance) ---
@st.cache_resource
def load_model():
    """Loads the trained Keras model from the .h5 file."""
    try:
        model = tf.keras.models.load_model("vae_decoder.h5")
        return model
    except FileNotFoundError:
        st.error("Model file 'vae_decoder.h5' not found. Please ensure it is in the same directory as the app script.")
        return None
    except Exception as e:
        st.error(f"An error occurred while loading the model: {e}")
        return None

model = load_model()

# --- Main App Logic ---
if model:
    st.header("🎨 Interactive Digit Generation")
    st.markdown("Choose a digit from the dropdown menu to see different generated versions.")

    # ==============================================================================
    # IMPORTANT: EDIT THIS DICTIONARY!
    # Replace these coordinates with the ones you found using the explorer tool.
    # This map tells the app where YOUR model generates each digit.
    # ==============================================================================
    latent_space_map = {
        # --- PASTE YOUR COORDINATES THAT YOU FOUND IN STEP 1 HERE ---
        0: [-0.50, 1.45],   # Example: Replace with your coordinates for '0'
        1: [-1.85, 1.80],   # Example: Replace with your coordinates for '1'
        2: [1.55, -0.40],   # Example: Replace with your coordinates for '2'
        3: [1.00, -1.50],   # ... and so on for all digits
        4: [-1.50, -0.80],
        5: [0.30, -1.00],
        6: [-0.50, -1.50],
        7: [-1.50, 0.50],
        8: [0.00, 0.00],
        9: [0.80, 0.80]
    }

    # --- User Interface Widgets ---
    col1, col2, col3 = st.columns(3)

    with col1:
        chosen_digit = st.selectbox(
            "**1. Choose a digit:**", options=list(range(10)), index=7
        )
    with col2:
        diversity_strength = st.slider(
            "**2. Select diversity:**", 0.0, 1.0, 0.25, 0.01,
            help="How much to vary the digit's shape. Lower is less diverse."
        )
    with col3:
        threshold = st.slider(
            "**3. Adjust clarity (threshold):**", 0.0, 1.0, 0.5, 0.01,
            help="Forces pixels to be black or white. Adjust for best clarity."
        )

    # --- Generation Logic ---
    if chosen_digit is not None:
        base_coord = np.array(latent_space_map[chosen_digit])
        noise = np.random.normal(scale=diversity_strength, size=(5, 2))
        z_batch_interactive = (base_coord + noise).astype(np.float32)
        
        # Predict all 5 images at once
        interactive_outputs = model.predict(z_batch_interactive)

        st.write(f"**Generated variations of the digit '{chosen_digit}':**")
        cols_interactive = st.columns(5)
        for i in range(5):
            with cols_interactive[i]:
                raw_image = interactive_outputs[i].reshape(8, 8)
                
                # Apply the thresholding fix to make the image sharp
                sharp_image = (raw_image > threshold).astype(float)
                
                fig, ax = plt.subplots(figsize=(3, 3))
                ax.imshow(sharp_image, cmap='binary', interpolation='nearest')
                ax.axis("off")
                st.pyplot(fig, use_container_width=True)
