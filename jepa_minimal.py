import torch
import torch.nn as nn

# Tiny JEPA: predict target embedding from masked context embedding

enc  = nn.Linear(16, 8)   # shared encoder (context + target)
pred = nn.Linear(8, 8)    # predictor
opt  = torch.optim.Adam(list(enc.parameters()) + list(pred.parameters()), lr=1e-3)

for step in range(200):
    x = torch.randn(32, 16)       # batch of inputs

    context = x.clone()
    context[:, 8:] = 0            # mask second half

    predicted = pred(enc(context))
    target    = enc(x).detach()   # stop gradient

    loss = ((predicted - target) ** 2).mean()
    opt.zero_grad()
    loss.backward()
    opt.step()

    if step % 50 == 0:
        print(f"step {step:3d}  loss={loss.item():.4f}")
