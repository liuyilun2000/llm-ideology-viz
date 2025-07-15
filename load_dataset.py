import os
import torch
from typing import List, Callable, Optional
from collections import Counter


import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns



from transformer_lens import HookedTransformer
from transformers import AutoTokenizer, AutoModel


from datasets import load_dataset

from utils import *



'''
# Load Hugging Face token from local file
with open('.hf_token', 'r') as f:
    HF_TOKEN = f.read().strip()

MODEL_NAME="mistralai/Mistral-7B-v0.3"
MODEL_NAME="meta-llama/Llama-2-7b-hf"
MODEL_NAME="meta-llama/Meta-Llama-3-8B"
MODEL_NAME="meta-llama/Llama-3.1-8B"
MODEL_NAME="meta-llama/Llama-3.2-3B"
MODEL_NAME="meta-llama/Llama-3.2-1B"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME,
    token=HF_TOKEN)
llama3_model = AutoModel.from_pretrained(MODEL_NAME,
    token=HF_TOKEN)
'''



dataset_name = "EN"

MODEL_NAME_LIST = [
    "Qwen/Qwen3-0.6B",
#    "Qwen/Qwen3-1.7B",
#    "Qwen/Qwen3-4B",
#    "Qwen/Qwen3-8B",
#    "mistralai/Mistral-7B-v0.3",
#    "meta-llama/Llama-2-7b-hf",
#    "meta-llama/Meta-Llama-3-8B",
#    "meta-llama/Llama-3.1-8B",
#    "meta-llama/Llama-3.2-1B",
#    "meta-llama/Llama-3.2-3B",
]

for MODEL_NAME in MODEL_NAME_LIST:
    print(AutoTokenizer.from_pretrained(MODEL_NAME,
        token=HF_TOKEN))
    print(AutoModel.from_pretrained(MODEL_NAME,
        token=HF_TOKEN))
    ACTIVATIONS_CACHE_DIR=f"data/cache/ParlaSent/{MODEL_NAME.split('/')[-1]}/{dataset_name}"
    if not os.path.exists(ACTIVATIONS_CACHE_DIR):
        os.makedirs(ACTIVATIONS_CACHE_DIR)
    
    df = load_parlasent([dataset_name])
    
    cache_activations(
        sentences=df['sentence'].tolist(),
        model_name=MODEL_NAME,
        output_dir=ACTIVATIONS_CACHE_DIR,
        activation_type="mlp",
        device="cuda" if torch.cuda.is_available() else "cpu",
        batch_size=100
    )


# Later, when you want to load the activations:
all_activations = load_activations(ACTIVATIONS_CACHE_DIR)
# Ensure all_activations is a numpy array
if isinstance(all_activations, torch.Tensor):
    all_activations = all_activations.cpu().numpy()


# Check if the number of activations matches the number of sentences
assert len(all_activations) == len(df), "Number of activations does not match number of sentences in DataFrame"


# Add the embeddings as a new column in the DataFrame
df['embedding'] = list(all_activations)



