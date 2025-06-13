from transformers import AutoTokenizer, AutoModel

import datetime
import pandas as pd
import numpy as np

import json
import xml.etree.ElementTree as ET

from tqdm.auto import tqdm

import plotly.express as px
import plotly.graph_objs as go

import torch

# Load Hugging Face token from a local file
with open('.hf_token', 'r') as f:
    HF_TOKEN = f.read().strip()


'''
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B", 
    cache_dir=TRANSFORMERS_CACHE_DIR, 
    token=HF_TOKEN)
llama3_model = AutoModel.from_pretrained("meta-llama/Meta-Llama-3-8B", 
    cache_dir=TRANSFORMERS_CACHE_DIR, 
    device_map='cpu',
    token=HF_TOKEN)
'''


from transformer_lens import HookedTransformer
model = HookedTransformer.from_pretrained('meta-llama/Meta-Llama-3-8B',
    cache_dir=TRANSFORMERS_CACHE_DIR)
torch.set_grad_enabled(False)
model.eval()


tmp = []

def store_function(activation, hook):
    tmp.append(torch.mean(activation[0], dim=0))

mlp_activation_filter = lambda name: ("mlp" in name) and ("hook_post" in name)

df_speech_filtered_concat = pd.read_pickle('df_speech_filtered_concat.pkl')

index = 0
ACTIVATIONS_CACHE_DIR = "/home/atuin/b207dd/b207dd11/test/DEU/activations/llama3-8b"

for sentence in tqdm(df_speech_filtered_concat['Text']):
    tmp = []
    model.run_with_hooks(
        sentence,
        return_type=None,
        fwd_hooks=[(mlp_activation_filter, store_function)],
    )
    res = torch.stack(tmp)
    torch.save(res.cpu(), ACTIVATIONS_CACHE_DIR+f'/{index}.pt')
    index += 1