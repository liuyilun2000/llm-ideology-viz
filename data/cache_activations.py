import argparse
import os
import torch
import pandas as pd
from tqdm.auto import tqdm
from transformer_lens import HookedTransformer


def cache_activations(
    texts, model_name, output_dir, activation_type="mlp", pooling_type="mean", device="cuda", batch_size=1, cache_dir=None
):
    # Load model
    model = HookedTransformer.from_pretrained(model_name, cache_dir=cache_dir)
    model.to(device)
    model.eval()
    torch.set_grad_enabled(False)

    if output_dir is None:
        output_dir = f"cache/activations/{model_name.split('/')[-1]}/{activation_type}_{pooling_type}"

    # Define hook/filter
    activations = []
    def store_function(activation, hook):
        if pooling_type == "mean":
            activations.append(torch.mean(activation[0], dim=0))
        elif pooling_type == "max":
            activations.append(torch.max(activation[0], dim=0))
        else:
            raise NotImplementedError("Only mean and max pooling supported for now.")

    if activation_type == "mlp":
        activation_filter = lambda name: ("mlp" in name) and ("hook_post" in name)
    else:
        raise NotImplementedError("Only MLP activations supported for now.")

    # Extract activations
    batch_idx = 0
    batch_activations = []
    for idx, text in enumerate(tqdm(texts, desc="Extracting activations")):
        activations.clear()
        model.run_with_hooks(
            text,
            return_type=None,
            fwd_hooks=[(activation_filter, store_function)],
        )
        batch_activations.append(torch.stack(activations))
        if len(batch_activations) == batch_size:
            res = torch.stack(batch_activations)
            torch.save(res.cpu(), os.path.join(output_dir, f"{batch_idx}.pt"))
            batch_idx += 1
            batch_activations = []
    if len(batch_activations) > 0:
        res = torch.stack(batch_activations)
        torch.save(res.cpu(), os.path.join(output_dir, f"{batch_idx}.pt"))

def main(args=None):
    parser = argparse.ArgumentParser(description="Extract embeddings from models.")
    parser.add_argument("--df_source", type=str, required=True, help="Input file (pickle/csv) with texts")
    parser.add_argument("--text_column", type=str, default="Text", help="Column name for texts")
    parser.add_argument("--model", type=str, required=True, help="Model name (e.g., meta-llama/Meta-Llama-3-8B)")
    parser.add_argument("--output_dir", type=str, default=None, help="Directory to save activations")
    parser.add_argument("--activation_type", type=str, default="mlp", help="Activation type (default: mlp)")
    parser.add_argument("--pooling_type", type=str, default="mean", help="Pooling type (mean/max, default: mean)")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda or cpu)")
    parser.add_argument("--cache_dir", type=str, default=None, help="Model cache directory")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size of saved activations")
    args = parser.parse_args() if args is None else args

    os.makedirs(args.output_dir, exist_ok=True)
    # Load data
    if args.df_source.endswith(".pkl"):
        df = pd.read_pickle(args.df_source)
    elif args.df_source.endswith(".csv"):
        df = pd.read_csv(args.df_source)
    else:
        raise ValueError("Input dataframe source file must be .pkl or .csv")

    texts = df[args.text_column].tolist()
    cache_activations(
        texts,
        model_name=args.model,
        output_dir=args.output_dir,
        activation_type=args.activation_type,
        pooling_type=args.pooling_type,
        device=args.device,
        batch_size=args.batch_size,
        cache_dir=args.cache_dir,
    )

if __name__ == "__main__":
    main() 