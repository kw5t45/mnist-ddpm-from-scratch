import torch
from torchvision import datasets, transforms
import deepinv
from tqdm import tqdm

batch_size = 32
image_size = 32  # resizing MNIST digits from 28x28 to 32x32 for convenience


transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.ToTensor(),
    transforms.Normalize((0.0,), (1.0,)),
])


train_loader = torch.utils.data.DataLoader(
    datasets.MNIST(root="./data", train=True, download=True, transform=transform),
    batch_size=batch_size,
    shuffle=True,
)

# data shape - should get 32, 1, 32, 32
data_iter = iter(train_loader)
images, labels = next(data_iter)
print(images.shape)


device = "cuda" if torch.cuda.is_available() else "cpu"

# diffusion parameters - similar to original ddpm paper
timesteps = 1000
beta_start = 1e-4
beta_end = 0.02

betas = torch.linspace(beta_start, beta_end, timesteps).to(device)
alphas = 1.0 - betas
alphas_cumprod = torch.cumprod(alphas, dim=0)
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)

# denoising model creation - simple unet implementation using deepinv library.
model = deepinv.models.DiffUNet(
    in_channels=1,       # grayscale MNIST images
    out_channels=1,      # predicting noise (same shape as input)
    pretrained=None      # we're training from scratch
).to(device)

#training params
lr = 1e-4
epochs = 5

optimizer = torch.optim.Adam(model.parameters(), lr=lr)
mse = deepinv.loss.MSE()

# model training

for epoch in tqdm(range(epochs)):
    seen = 0
    model.train()
    for data, _ in tqdm(train_loader):
        seen += 32
        if seen >= 1000: # for early stopping
            break
        imgs = data.to(device)
        noise = torch.randn_like(imgs)
        t = torch.randint(0, timesteps, (imgs.size(0),), device=device)

        noised_imgs = (
            sqrt_alphas_cumprod[t, None, None, None] * imgs
            + sqrt_one_minus_alphas_cumprod[t, None, None, None] * noise
        )

        optimizer.zero_grad()
        estimated_noise = model(noised_imgs, t, type_t="timestep")
        loss = mse(estimated_noise, noise).mean()
        print(loss)
        loss.backward()
        optimizer.step()

torch.save(
    model.state_dict(),
    "trained_ddpm.pth",
)










