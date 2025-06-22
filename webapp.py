import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# --- Page Setup ---
st.set_page_config(page_title="🧠 Digit Generator", layout="wide")
st.title("🧠 Handwritten Digit Generator")
st.markdown("This app uses a Keras VAE decoder to generate handwritten digits from a 2D latent space.")

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
    # --- Section 1: Random Generation ---
    st.subheader("🎲 Completely Random Samples")
    st.markdown("These digits are generated from random points in the latent space. Click the button to get new ones.")

    if st.button("Generate New Random Digits"):
        z_batch = np.random.normal(size=(5, 2)).astype(np.float32)
        outputs = model.predict(z_batch)

        cols = st.columns(5)
        for i in range(5):
            with cols[i]:
                image = outputs[i].reshape(8, 8)
                fig, ax = plt.subplots(figsize=(3, 3)) # Slightly larger figure
                # Use 'binary' colormap (black on white) and 'nearest' interpolation for a sharp, pixelated look
                ax.imshow(image, cmap='binary', interpolation='nearest') # <-- CHANGE HERE
                ax.axis("off")
                st.pyplot(fig, use_container_width=True)

    # --- Visual Separator ---
    st.divider()

    # --- Section 2: Interactive Generation (User Chooses a Digit) ---
    st.subheader("🎨 Interactive Digit Generation")
    st.markdown("Choose a digit and see how small changes in the latent space create diverse variations.")

    latent_space_map = {
        0: [-0.6, 1.4], 1: [-1.8, 1.8], 2: [1.5, -0.5], 3: [1.0, -1.5], 4: [-1.5, -0.8],
        5: [0.3, -1.0], 6: [-0.5, -1.5], 7: [-1.5, 0.5], 8: [0.0, 0.0], 9: [0.8, 0.8]
    }

    col1, col2 = st.columns([1, 2])

    with col1:
        chosen_digit = st.selectbox(
            "**Choose a digit to generate:**", options=list(range(10)), index=7
        )
    with col2:
        diversity_strength = st.slider(
            "**Select diversity (how much to vary the digit):**", 0.0, 1.0, 0.35, 0.05
        )

    if chosen_digit is not None:
        base_coord = np.array(latent_space_map[chosen_digit])
        noise = np.random.normal(scale=diversity_strength, size=(5, 2))
        z_batch_interactive = (base_coord + noise).astype(np.float32)
        interactive_outputs = model.predict(z_batch_interactive)

        st.write(f"**Generated variations of the digit '{chosen_digit}':**")
        cols_interactive = st.columns(5)
        for i in range(5):
            with cols_interactive[i]:
                image = interactive_outputs[i].reshape(8, 8)
                fig, ax = plt.subplots(figsize=(3, 3)) # Slightly larger figure
                # Apply the same fix here for sharp, clear images
                ax.imshow(image, cmap='binary', interpolation='nearest') # <-- CHANGE HERE
                ax.axis("off")
                st.pyplot(fig, use_container_width=True)
