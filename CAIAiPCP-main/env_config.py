import os
import torch

# DEVICE
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# HuggingFace Model Configuration
HUGGINGFACE_MODEL_NAME = "HuggingFaceTB/SmolLM3-3B"
# env_config.py
MCP_SERVER_URL = os.environ.get("MCP_SERVER_URL", "http://localhost:8000/sse")

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "AIzaSyBA0NX7FNwwT-tsgiozGAS8KYvPt97u2p4")



