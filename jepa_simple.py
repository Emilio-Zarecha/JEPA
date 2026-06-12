import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# JEPA: Joint Embedding Predictive Architecture
#
# Core idea:
#   - Context encoder  : encodes a masked/partial view, updated by gradient
#   - Target encoder   : encodes the full view, updated by EMA (stop gradient)
#   - Predictor        : maps context embedding -> predicted target embedding
#   - Loss             : MSE between prediction and target embedding (in latent space)

EMBED_DIM  = 64
INPUT_DIM  = 128
EMA_DECAY  = 0.99
LR         = 1e-3
STEPS      = 500
BATCH      = 32


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(INPUT_DIM, 256),
            nn.ReLU(),
            nn.Linear(256, EMBED_DIM),
        )

    def forward(self, x):
        return self.net(x)


class Predictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(EMBED_DIM, 256),
            nn.ReLU(),
            nn.Linear(256, EMBED_DIM),
        )

    def forward(self, x):
        return self.net(x)


def ema_update(online: nn.Module, target: nn.Module, decay: float):
    """Copy online -> target using exponential moving average."""
    with torch.no_grad():
        for p_online, p_target in zip(online.parameters(), target.parameters()):
            p_target.data = decay * p_target.data + (1 - decay) * p_online.data


def make_batch():
    """Synthetic data: context is a noisy/masked version of the target input."""
    target_input  = torch.randn(BATCH, INPUT_DIM)
    context_input = target_input + 0.1 * torch.randn(BATCH, INPUT_DIM)
    context_input[:, INPUT_DIM // 2:] = 0.0  # mask second half
    return context_input, target_input


context_enc = Encoder()
target_enc  = Encoder()
predictor   = Predictor()

# Target encoder starts as a copy of context encoder
target_enc.load_state_dict(context_enc.state_dict())

optimizer = optim.Adam(list(context_enc.parameters()) + list(predictor.parameters()), lr=LR)
loss_fn   = nn.MSELoss()

for step in range(1, STEPS + 1):
    context_input, target_input = make_batch()

    # Forward
    context_embed = context_enc(context_input)
    predicted     = predictor(context_embed)

    with torch.no_grad():
        target_embed = target_enc(target_input)   # stop gradient: no backprop here

    loss = loss_fn(predicted, target_embed)

    # Backward (only context_enc and predictor receive gradients)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # EMA update for target encoder
    ema_update(context_enc, target_enc, EMA_DECAY)

    if step % 100 == 0:
        print(f"step {step:4d}  loss={loss.item():.4f}")

print("Done.")

# --- Visualization: PCA of embeddings after training ---

N_VIS = 256
context_input, target_input = torch.randn(N_VIS, INPUT_DIM), torch.randn(N_VIS, INPUT_DIM)
context_input_masked = context_input.clone()
context_input_masked[:, INPUT_DIM // 2:] = 0.0

with torch.no_grad():
    ctx_emb  = context_enc(context_input_masked).numpy()
    pred_emb = predictor(context_enc(context_input_masked)).numpy()
    tgt_emb  = target_enc(target_input).numpy()

pca = PCA(n_components=2).fit(np.concatenate([ctx_emb, pred_emb, tgt_emb]))
ctx_2d  = pca.transform(ctx_emb)
pred_2d = pca.transform(pred_emb)
tgt_2d  = pca.transform(tgt_emb)

fig, axes = plt.subplots(1, 3, figsize=(14, 4))
fig.suptitle("JEPA Embeddings (PCA)", fontsize=14)

for ax, pts, label, color in zip(
    axes,
    [ctx_2d, pred_2d, tgt_2d],
    ["Context Encoder", "Predictor Output", "Target Encoder"],
    ["steelblue", "darkorange", "seagreen"],
):
    ax.scatter(pts[:, 0], pts[:, 1], alpha=0.4, s=15, color=color)
    ax.set_title(label)
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")

plt.tight_layout()
plt.savefig("embeddings.png", dpi=150)
print("Saved embeddings.png")
