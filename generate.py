# === generate.py ===
import torch
import torch.nn as nn
import tiktoken

# Reuse the same model classes
from train import TransformerLanguageModel

# === Generate Function ===
def load_model_and_generate(prompt_text, checkpoint_path="transformer_epoch0.pt", max_tokens=50):
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(device)

    enc = tiktoken.get_encoding("gpt2")
    model = TransformerLanguageModel(
        vocab_size=50257, embed_dim=512, num_heads=8, num_layers=6, block_size=128
    ).to(device)

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    # Encode input
    context_tokens = enc.encode(prompt_text)
    context = torch.tensor([context_tokens], dtype=torch.long).to(device)

    # Generate
    generated = model.generate(context, max_new_tokens=max_tokens)
    output = enc.decode(generated[0].tolist())
    return output

# === Main Execution ===
if __name__ == "__main__":
    prompt = input("Enter your sentence prompt: ")
    output = load_model_and_generate(prompt)
    print("\nGenerated:")
    print(output)