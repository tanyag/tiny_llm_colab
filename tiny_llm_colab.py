# tiny_llm_colab.py
import torch
import torch.nn as nn
import torch.nn.functional as F

# Sample dataset
text = "hello world. this is a test of a tiny language model. " * 100
chars = sorted(list(set(text)))
vocab_size = len(chars)

# Mappings
char_to_idx = {ch: i for i, ch in enumerate(chars)}
idx_to_char = {i: ch for i, ch in enumerate(chars)}

def encode(s): return [char_to_idx[c] for c in s]
def decode(indices): return ''.join([idx_to_char[i] for i in indices])

data = torch.tensor(encode(text), dtype=torch.long)

# Batching
block_size = 8
batch_size = 4

def get_batch():
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    return x, y

# Model
class TinyLLM(nn.Module):
    def __init__(self, vocab_size, n_embed=32):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, n_embed)
        self.fc = nn.Linear(n_embed, vocab_size)

    def forward(self, idx):
        x = self.embed(idx)
        logits = self.fc(x)
        return logits

model = TinyLLM(vocab_size)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Training
for step in range(500):
    x, y = get_batch()
    logits = model(x)
    loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if step % 100 == 0:
        print(f"Step {step}, Loss: {loss.item():.4f}")

# Generation
context = torch.tensor([[char_to_idx['h']]], dtype=torch.long)
generated = context
model.eval()
for _ in range(100):
    logits = model(generated)
    probs = F.softmax(logits[:, -1, :], dim=-1)
    next_token = torch.multinomial(probs, num_samples=1)
    generated = torch.cat((generated, next_token), dim=1)

print("\nGenerated Text:")
print(decode(generated[0].tolist()))
