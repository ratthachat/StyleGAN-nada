%%writefile /kaggle/working/stylegan_nada/ZSSGAN/utils/training_utils.py
import torch
import math
import random
import os

from torch.utils.data import Dataset
from pytorch_lightning.callbacks import Callback

class NADACallback(Callback):
    def __init__(self, output_dir, output_interval=0, save_interval=0 ):
        self.current_epoch = 0
        self.output_interval = output_interval
        self.save_interval = save_interval
        self.sample_dir = os.path.join(output_dir, "sample")
        self.ckpt_dir = os.path.join(output_dir, "checkpoint")
        
        os.makedirs(self.sample_dir, exist_ok=True)
        os.makedirs(self.ckpt_dir, exist_ok=True)
        
    def on_train_epoch_end(self, trainer, net):
        self.current_epoch += 1
        if self.output_interval > 0 and self.current_epoch % self.output_interval  == 0:
            net.generate_samples_jupyter(self.sample_dir, use_fixed_z=True, prefix_str="dst", postfix_num=self.current_epoch)
            
        if (self.save_interval > 0) and (self.current_epoch > 0) and self.current_epoch % self.save_interval  == 0:
            print('saving checkpoint at iteration', self.current_epoch)
            torch.save(
                {
                "g_ema": net.generator_trainable.generator.state_dict(),
                "g_optim": net.optimizers().state_dict(),
                },
                f"{self.ckpt_dir}/{str(self.current_epoch).zfill(6)}.pt",
            )

class NADADataset(Dataset):
    def __init__(self, mixing):
        self.x = 1
        self.mixing = mixing
        
    def __len__(self):
        return 64 # Keep DataLoader busy for a while

    def __getitem__(self, idx):
        z = mixing_noise(1, 512, 
                         self.mixing,
                         device='cpu' #'cuda:0'
                        ) # generate a list of single example [(1,512)]
        y = -1 # dummy, don't use
        return [z[0].squeeze()], y
            
def make_noise(batch, latent_dim, n_noise, device):
    if n_noise == 1:
        return torch.randn(batch, latent_dim, device=device)

    noises = torch.randn(n_noise, batch, latent_dim, device=device).unbind(0)

    return noises


def mixing_noise(batch, latent_dim, prob, device):
    if prob > 0 and random.random() < prob:
        return make_noise(batch, latent_dim, 2, device)

    else:
        return [make_noise(batch, latent_dim, 1, device)]

def g_path_regularize(fake_img, latents, mean_path_length, decay=0.01):
    noise = torch.randn_like(fake_img) / math.sqrt(
        fake_img.shape[2] * fake_img.shape[3]
    )
    grad, = torch.autograd.grad(
        outputs=(fake_img * noise).sum(), inputs=latents, create_graph=True
    )
    path_lengths = torch.sqrt(grad.pow(2).sum(2).mean(1))

    path_mean = mean_path_length + decay * (path_lengths.mean() - mean_path_length)

    path_penalty = (path_lengths - path_mean).pow(2).mean()

    return path_penalty, path_mean.detach(), path_lengths
