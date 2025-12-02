import os
import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "AIzaSyBA0NX7FNwwT-tsgiozGAS8KYvPt97u2p4")