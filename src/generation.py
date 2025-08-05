import torch
import deepinv
import matplotlib.pyplot as plt
from tqdm import tqdm


#loading model
device = "cuda" if torch.cuda.is_available() else "cpu"
model = deepinv.models.DiffUNet(in_channels=1, out_channels=1, pretrained=None).to(device)
model.load_state_dict(torch.load(rf"trained_ddpm.pth", map_location=device))
model.eval()

# same training params
timesteps = 1000
beta_start = 1e-4
beta_end = 0.02

betas = torch.linspace(beta_start, beta_end, timesteps).to(device)
alphas = 1.0 - betas
alphas_cumprod = torch.cumprod(alphas, dim=0)
sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)

# gaussian noise
shape = (1, 1, 32, 32)  # 1 sample, grayscale, 32x32
x = torch.randn(shape).to(device)

# main sampling loop
snapshots = [] # for seeing process

for t in tqdm(reversed(range(timesteps))):
    t_tensor = torch.full((1,), t, device=device, dtype=torch.long)
    with torch.no_grad():
        predicted_noise = model(x, t_tensor, type_t="timestep")

    alpha = alphas[t]
    alpha_cumprod = alphas_cumprod[t]
    beta = betas[t]

    # predict x_{t-1}
    x = (
        (1 / torch.sqrt(alpha)) *
        (x - (beta / torch.sqrt(1 - alpha_cumprod)) * predicted_noise)
    )

    # adding brownian motion -  noise if not final step
    if t > 0:
        noise = torch.randn_like(x)
        x += torch.sqrt(beta) * noise

    # saving snapshot every 100 steps
    if t % 100 == 0:
        img = x.squeeze().detach().cpu().clamp(0, 1).numpy()
        snapshots.append(img)

# visualizing all 10 steps
fig, axes = plt.subplots(2, 5, figsize=(12, 5))
for i, ax in enumerate(axes.flat):
    ax.imshow(snapshots[i], cmap="gray")
    ax.set_title(f"t={timesteps - i * 100}")
    ax.axis("off")

plt.tight_layout()
plt.show()

