from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM

import datetime
import pandas as pd
import numpy as np

import itertools

import json
import xml.etree.ElementTree as ET

from tqdm.auto import tqdm

import plotly.express as px
import plotly.graph_objs as go

import torch

from torch.utils.data import DataLoader, TensorDataset, random_split
import torch.nn as nn
import torch.optim as optim

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, accuracy_score, f1_score, r2_score, classification_report

import plotly.express as px

import importlib
import warnings
from collections import defaultdict
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)




TRANSFORMERS_CACHE_DIR = "/home/atuin/b207dd/b207dd11/cache/huggingface/transformers"

# Load Hugging Face token from a local file
with open('.hf_token', 'r') as f:
    HF_TOKEN = f.read().strip()

model_name = "meta-llama/Meta-Llama-3-8B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name, 
    cache_dir=TRANSFORMERS_CACHE_DIR, 
    token=HF_TOKEN)

llama3_chat_model = AutoModelForCausalLM.from_pretrained(model_name, 
    cache_dir=TRANSFORMERS_CACHE_DIR, 
    device_map='auto',
    torch_dtype=torch.bfloat16,
    token=HF_TOKEN)

model = llama3_chat_model
tokenizer.pad_token_id = tokenizer.eos_token_id
model.generation_config.pad_token_id = tokenizer.pad_token_id

model.eval()
terminators = [
    tokenizer.eos_token_id,
    tokenizer.convert_tokens_to_ids("<|eot_id|>")
]


# List of topics
topics = [
    "Wirtschaftspolitik", "Sozialpolitik", "Bildungspolitik", "Gesundheitspolitik", 
    "Umweltpolitik", "Außenpolitik", "Verteidigungspolitik", "Innenpolitik", 
    "Justizpolitik", "Verkehrspolitik", "Familienpolitik", "Migrationspolitik", 
    "Energiepolitik", "Digitalisierung", "Kulturpolitik", "Landwirtschaftspolitik", 
    "Wohnungspolitik", "Verbraucherschutz", "Gesellschaftspolitik", "Europapolitik"
]


# List of possible stances for each axis
axes_libertarian  = {0: "Neutral", 1: "Libertär", -1: "Restriktiv"}
axes_collectivist = {0: "Neutral", 1: "Kollektivistisch", -1: "Individualistisch"}
axes_progressive  = {0: "Neutral", 1: "Progressiv", -1: "Konservativ"}

def generate_system_prompt(libertarian_stance, collectivist_stance, progressive_stance):
    libertarian_dict = {
        0: "neutral zwischen libertär und restriktiv",
        1: "libertär statt restriktiv",
        -1: "restriktiv statt libertär"
    }
    collectivist_dict = {
        0: "neutral zwischen kollektivistisch und individualistisch",
        1: "kollektivistisch statt individualistisch",
        -1: "individualistisch statt kollektivistisch"
    }
    progressive_dict = {
        0: "neutral zwischen progressiv und konservativ",
        1: "progressiv statt konservativ", 
        -1: "konservativ statt progressiv"
    }
    return f"Sie sind ein Mitglied des Deutschen Bundestages. Sie sind {libertarian_dict[libertarian_stance]}. Sie sind {collectivist_dict[collectivist_stance]}. Sie sind {progressive_dict[progressive_stance]}. Erinnern Sie sich daran, wer Sie sind, wenn Sie in der Öffentlichkeit sprechen."



# Define function to get response for a topic and ideology
def get_response(topic, system_prompt):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Bitte äußern Sie kurz Ihre Meinung zu {topic}."}
    ]    
    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)
    outputs = model.generate(
        input_ids,
        max_new_tokens=512,
        eos_token_id=terminators,
        do_sample=True,
        temperature=0.6,
        top_p=0.9,
    )
    response = outputs[0][input_ids.shape[-1]:]
    return tokenizer.decode(response, skip_special_tokens=True)


responses = {}
combinations = list(itertools.product(topics, axes_libertarian.items(), axes_collectivist.items(), axes_progressive.items()))
for (topic, (libertarian_k, libertarian_v), (collectivist_k, collectivist_v), (progressive_k, progressive_v)) in tqdm(combinations):
    system_prompt = generate_system_prompt(libertarian_k, collectivist_k, progressive_k)
    response = get_response(topic, system_prompt)
    key = (libertarian_v, collectivist_v, progressive_v)
    if key not in responses:
        responses[key] = {}
    responses[key][topic] = response
    #
    json_compatible_responses = {
        f"libertarian-{libertarian_stance}_collectivist-{collectivist_stance}_progressive-{progressive_stance}": topics_responses
        for (libertarian_stance, collectivist_stance, progressive_stance), topics_responses in responses.items()
    }
    # Save the responses to a JSON file
    with open("/home/hpc/b207dd/b207dd11/test/spin-politics/responses.json", "w", encoding="utf-8") as f:
        json.dump(json_compatible_responses, f, ensure_ascii=False, indent=4)




# Create DataFrame from responses
flattened_data = []

for (libertarian_v, collectivist_v, progressive_v), topics_responses in responses.items():
    for topic, response in topics_responses.items():
        flattened_data.append({
            "Libertarian": libertarian_v,
            "Collectivist": collectivist_v,
            "Progressive": progressive_v,
            "Topic": topic,
            "Response": response
        })

df = pd.DataFrame(flattened_data)
df.to_pickle("/home/hpc/b207dd/b207dd11/test/spin-politics/responses.pkl")








df = pd.read_pickle("/home/hpc/b207dd/b207dd11/test/spin-politics/responses.pkl")


from transformer_lens import HookedTransformer
model = HookedTransformer.from_pretrained('meta-llama/Meta-Llama-3-8B',
    cache_dir=TRANSFORMERS_CACHE_DIR)
torch.set_grad_enabled(False)
model.eval()

def store_function(activation, hook):
    tmp.append(torch.mean(activation[0], dim=0))

mlp_activation_filter = lambda name: ("mlp" in name) and ("hook_post" in name)

index = 0
ACTIVATIONS_CACHE_DIR = "/home/atuin/b207dd/b207dd11/test/DEU/standard_sentences/llama3-8b"
for sentence in tqdm(df['Response']):
    tmp = []
    model.run_with_hooks(
        sentence,
        return_type=None,
        fwd_hooks=[(mlp_activation_filter, store_function)],
    )
    res = torch.stack(tmp)
    torch.save(res.cpu(), ACTIVATIONS_CACHE_DIR+f'/{index}.pt')
    index += 1




from transformer_lens import HookedTransformer
model = HookedTransformer.from_pretrained('meta-llama/Llama-2-7b-hf',
    cache_dir=TRANSFORMERS_CACHE_DIR)
torch.set_grad_enabled(False)
model.eval()

def store_function(activation, hook):
    tmp.append(torch.mean(activation[0], dim=0))

mlp_activation_filter = lambda name: ("mlp" in name) and ("hook_post" in name)

index = 0
ACTIVATIONS_CACHE_DIR = "/home/atuin/b207dd/b207dd11/test/DEU/standard_sentences/llama2-7b"
for sentence in tqdm(df['Response']):
    tmp = []
    model.run_with_hooks(
        sentence,
        return_type=None,
        fwd_hooks=[(mlp_activation_filter, store_function)],
    )
    res = torch.stack(tmp)
    torch.save(res.cpu(), ACTIVATIONS_CACHE_DIR+f'/{index}.pt')
    index += 1







import os
import torch

# Path to the directory containing the saved tensor files
ACTIVATIONS_CACHE_DIR = "/home/atuin/b207dd/b207dd11/test/DEU/standard_sentences/llama3-8b"

# Function to load embeddings from a tensor file
def load_embedding(index):
    file_path = os.path.join(ACTIVATIONS_CACHE_DIR, f'{index}.pt')
    return torch.load(file_path)

# Load all embeddings and add them to the DataFrame
embeddings = []
for index in tqdm(range(len(df))):
    embeddings.append(load_embedding(index).numpy())

df['Embedding'] = embeddings

# Save the updated DataFrame with embeddings to a new pickle file
df.to_pickle("/home/hpc/b207dd/b207dd11/test/spin-politics/standard_text_with_embeddings.pkl")

import ace_tools as tools; tools.display_dataframe_to_user(name="Responses with Embeddings", dataframe=df)
