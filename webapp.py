import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

# Streamlit config
st.set_page_config(page_title="🧠 Digit Generator (PyTorch)", layout="wide")
st.title("🧠 Handwritten Digit Generator")
st.markdown("This app uses a PyTorch-based VAE decoder to generate handwritten digits.")

# Define the decoder architecture (example architecture)
class Decoder(nn.Module):
    def __init__(self, latent_dim=2):
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),  # Output shape (8x8 = 64)
            nn.Sigmoid()  # To map pixel values between 0 and 1
        )

    def forward(self, z):
        return self.decoder(z)

# Load the model (cached)
@st.cache_resource
def load_model():
    model = Decoder()
    model.load_state_dict(torch.load("vae_decoder.pth", map_location=torch.device("cpu")))
    model.eval()
    return model

decoder = load_model()

# UI
st.subheader("🖼️ Generated Digit Variations")
cols = st.columns(5)

for i in range(5):
    with cols[i]:
        # Sample random latent vector
        z = torch.randn(1, 2)  # 2D latent space

        # Generate image
        output = decoder(z).detach().numpy()
        image = output.reshape(8, 8)

        # Plot image
        fig, ax = plt.subplots()
        ax.imshow(image, cmap="gray")
        ax.axis("off")
        st.pyplot(fig)
