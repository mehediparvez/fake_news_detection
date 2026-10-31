
import sys
import torch
import pandas as pd
import sklearn
import streamlit

print("Python version:", sys.version)
print("PyTorch version:", torch.__version__)
print("Pandas version:", pd.__version__)
print("Scikit-learn version:", sklearn.__version__)
print("Streamlit version:", streamlit.__version__)

print("CUDA available:", torch.cuda.is_available())

print("\nAll packages are working!")
