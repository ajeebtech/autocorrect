# === train.py ===
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import tiktoken

# === Positional Encoding ===
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

# === Transformer Block ===
class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.ReLU(),
            nn.Linear(4 * embed_dim, embed_dim)
        )
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x, mask=None):
        attn_output, _ = self.attention(x, x, x, attn_mask=mask)
        x = self.norm1(x + attn_output)
        ff_output = self.ff(x)
        x = self.norm2(x + ff_output)
        return x

# === Full Transformer Language Model ===
class TransformerLanguageModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, num_layers, block_size):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_encoding = PositionalEncoding(embed_dim, max_len=block_size)
        self.transformer_blocks = nn.ModuleList(
    [TransformerBlock(embed_dim, num_heads) for _ in range(num_layers)])
        self.output_layer = nn.Linear(embed_dim, vocab_size)

    def forward(self, idx, mask=None):
        token_embeddings = self.token_embedding(idx)  # Embedding tokens
        x = self.position_encoding(token_embeddings)  # Add positional encoding
        for block in self.transformer_blocks:  # Pass through all Transformer blocks
            x = block(x, mask=mask)  # Pass mask along with the input if available
        logits = self.output_layer(x)  # Output layer
        return logits


    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0):
        for _ in range(max_new_tokens):
            input_condensed = idx[:, -128:] if idx.size(1) > 128 else idx
            logits = self.forward(input_condensed)
            logits = logits[:, -1, :] / temperature
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_token), dim=1)
        return idx

# === Dataset ===
class TextDataset(torch.utils.data.Dataset):
    def __init__(self, file_path, block_size=128):
        enc = tiktoken.get_encoding("gpt2")
        with open(file_path, "r", encoding="utf-8") as f:
            data = f.read()
        self.tokens = enc.encode(data)
        self.block_size = block_size

    def __len__(self):
        return len(self.tokens) - self.block_size

    def __getitem__(self, idx):
        x = torch.tensor(self.tokens[idx:idx + self.block_size], dtype=torch.long)
        y = torch.tensor(self.tokens[idx + 1:idx + self.block_size + 1], dtype=torch.long)
        return x, y

# === Training ===
def train():
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'

    dataset = TextDataset("big.txt", block_size=128)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = TransformerLanguageModel(
        vocab_size=50257, embed_dim=512, num_heads=8, num_layers=6, block_size=128
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    criterion = nn.CrossEntropyLoss()

    import tqdm

    for epoch in range(5):
        model.train()
        total_loss = 0
        progress_bar = tqdm.tqdm(loader, desc=f"Epoch {epoch+1}/5")
        
        for step, (x, y) in enumerate(progress_bar):
            x, y = x.to(device), y.to(device)
            logits = model(x)
            B, T, V = logits.size()
            loss = criterion(logits.view(B * T, V), y.view(B * T))

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            
            # Update progress bar description with current loss
            progress_bar.set_postfix(loss=f"{loss.item():.4f}")
            
            if step % 100 == 0:
                print(f"Epoch {epoch+1} Step {step} Loss {loss.item():.4f}")

        torch.save(model.state_dict(), f"transformer_epoch{epoch}.pt")
        print(f"Epoch {epoch+1} completed. Avg Loss: {total_loss / len(loader):.4f}")

if __name__ == "__main__":
    train()
