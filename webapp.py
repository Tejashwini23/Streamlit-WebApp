import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# --- Page Setup ---
st.set_page_config(page_title="🧠 Digit Explorer", layout="wide")
st.title("🧠 Latent Space Digit Explorer")
st.markdown("""
This app lets you explore the 'mind' of a generative model.
Since the generated digits were blurry, this tool helps you find the precise coordinates in the latent space where clear digits are formed.
""")

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
    st.header("🎨 Interactive Latent Space Explorer")
    st.info("💡 **How to use:** Slowly move the sliders to explore the 2D space. Watch the image change and try to find the 'sweet spots' where clear digits appear.")

    # --- Create two columns for the sliders and the output image ---
    col1, col2 = st.columns([1, 2]) # Make the image column wider

    with col1:
        st.subheader("Controls")
        # Create sliders for the user to pick the exact coordinates
        z_x = st.slider(
            "Latent Variable 'x'",
            min_value=-3.0,
            max_value=3.0,
            value=0.0, # Start at the center
            step=0.05
        )
        z_y = st.slider(
            "Latent Variable 'y'",
            min_value=-3.0,
            max_value=3.0,
            value=0.0, # Start at the center
            step=0.05
        )
        
        # Add a threshold slider for even more control
        threshold = st.slider(
            "Clarity Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.5, # 0.5 is a standard choice
            step=0.01,
            help="Forces pixels to be black or white. Higher values mean more black."
        )


    # --- Generation and Display Logic ---
    with col2:
        st.subheader("Generated Image")

        # 1. Create the latent vector from the slider values
        z = np.array([[z_x, z_y]], dtype=np.float32)

        # 2. Generate the output from the model
        output = model.predict(z)

        # 3. Reshape the raw output
        raw_image = output[0].reshape(8, 8)
        
        # 4. **CRUCIAL FIX: Apply Thresholding**
        # Any pixel value above the threshold becomes 1 (white), and below becomes 0 (black).
        # This removes all blurriness and grayness.
        sharp_image = (raw_image > threshold).astype(float)

        # 5. Display the sharp, high-contrast image
        fig, ax = plt.subplots(figsize=(6, 6)) # Make the plot large
        
        # Use 'binary' colormap and 'nearest' interpolation for a sharp, pixelated look
        ax.imshow(sharp_image, cmap='binary', interpolation='nearest')
        ax.axis("off") # Hide the x and y axes

        st.pyplot(fig)
        st.write(f"Current Coordinates: `(x={z_x:.2f}, y={z_y:.2f})`")

# --- Explanation Section ---
st.divider()
st.subheader("Why Were the Images So Blurry?")
st.markdown("""
*   **Model's Nature:** VAEs learn to create "average" versions of digits. This often results in blurry outputs, especially with low-resolution (8x8) images.
*   **Wrong Coordinates:** The previous version of the app used a hard-coded map of coordinates for each digit. That map was a guess and did not match how **your specific model** learned to organize digits. This explorer tool lets you find the correct coordinates for your model.
*   **No Post-Processing:** The model outputs shades of gray. By adding a **threshold**, we force every pixel to be either pure black or white, creating a much sharper final image.
""")
