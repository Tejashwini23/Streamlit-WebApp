import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# --- Page Setup ---
st.set_page_config(page_title="🧠 Digit Generator", layout="wide")
st.title("🧠 Handwritten Digit Generator")
st.markdown("This app uses a Keras VAE decoder to generate handwritten digits from a 2D latent space.")

# --- Load the Model (Cached for Performance) ---
# This function will only run once, and the result will be stored.
@st.cache_resource
def load_model():
    """Loads the trained Keras model from the .h5 file."""
    try:
        # Assumes 'vae_decoder.h5' is in the same folder as this script.
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
# Only proceed if the model was loaded successfully.
if model:
    # --- Section 1: Random Generation ---
    st.subheader("🎲 Completely Random Samples")
    st.markdown("These digits are generated from random points in the latent space. Click the button to get new ones.")

    if st.button("Generate New Random Digits"):
        # Generate a BATCH of 5 random latent vectors for efficiency
        z_batch = np.random.normal(size=(5, 2)).astype(np.float32)
        outputs = model.predict(z_batch)

        cols = st.columns(5)
        for i in range(5):
            with cols[i]:
                # Reshape the output to an 8x8 image (adjust if your model outputs a different size)
                image = outputs[i].reshape(8, 8)
                fig, ax = plt.subplots(figsize=(2, 2))
                ax.imshow(image, cmap='gray')
                ax.axis("off")
                st.pyplot(fig, use_container_width=True)

    # --- Visual Separator ---
    st.divider()

    # --- Section 2: Interactive Generation (User Chooses a Digit) ---
    st.subheader("🎨 Interactive Digit Generation")
    st.markdown("Choose a digit and see how small changes in the latent space create diverse variations.")

    # This is the key part: A pre-defined map of where each digit is located
    # in the 2D latent space. These are our "best guess" coordinates for each digit.
    # The user NEVER sees these coordinates.
    latent_space_map = {
        0: [-0.6, 1.4],   # Guess for '0'
        1: [-1.8, 1.8],   # Guess for '1'
        2: [1.5, -0.5],   # Guess for '2'
        3: [1.0, -1.5],   # Guess for '3'
        4: [-1.5, -0.8],  # Guess for '4'
        5: [0.3, -1.0],   # Guess for '5'
        6: [-0.5, -1.5],  # Guess for '6'
        7: [-1.5, 0.5],   # Guess for '7'
        8: [0.0, 0.0],    # Guess for '8' (often near the center)
        9: [0.8, 0.8]     # Guess for '9'
    }

    # --- User Interface Widgets ---
    col1, col2 = st.columns([1, 2]) # Make the second column wider

    with col1:
        # A simple dropdown for the user to choose a digit.
        chosen_digit = st.selectbox(
            "**Choose a digit to generate:**",
            options=list(range(10)),
            index=7 # Default to showing '7'
        )

    with col2:
        # A slider for the user to control the amount of variation.
        diversity_strength = st.slider(
            "**Select diversity (how much to vary the digit):**",
            min_value=0.0,
            max_value=1.0,
            value=0.35, # A good starting point
            step=0.05
        )

    # --- Generation Logic for the chosen digit ---
    if chosen_digit is not None:
        # 1. Get the base coordinate for the chosen digit from our hidden map.
        base_coord = np.array(latent_space_map[chosen_digit])

        # 2. Generate a batch of 5 noise vectors. The scale is controlled by the slider.
        noise = np.random.normal(scale=diversity_strength, size=(5, 2))

        # 3. Add the noise to the base coordinate to get 5 slightly different points.
        z_batch_interactive = (base_coord + noise).astype(np.float32)

        # 4. Predict all 5 images at once.
        interactive_outputs = model.predict(z_batch_interactive)

        # 5. Display the 5 generated variations.
        st.write(f"**Generated variations of the digit '{chosen_digit}':**")
        cols_interactive = st.columns(5)
        for i in range(5):
            with cols_interactive[i]:
                image = interactive_outputs[i].reshape(8, 8)
                fig, ax = plt.subplots(figsize=(2, 2))
                ax.imshow(image, cmap='gray')
                ax.axis("off")
                st.pyplot(fig, use_container_width=True)
