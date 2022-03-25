import sys
import os
sys.path.insert(0, os.path.abspath('../'))

import torch
import torchvision.transforms as transforms

import numpy as np
import copy

from functools import partial

from ZSSGAN.model.sg2_model import Generator, Discriminator
from ZSSGAN.criteria.clip_loss import CLIPLoss
from ZSSGAN.utils.training_utils import mixing_noise
from ZSSGAN.utils.file_utils import save_images

from IPython.display import display
import os
from PIL import Image
import pytorch_lightning as pl

def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag

class SG2Generator(torch.nn.Module):
    def __init__(self, checkpoint_path, latent_size=512, map_layers=8, img_size=256, channel_multiplier=2, device='cuda:0'):
        super(SG2Generator, self).__init__()

        self.generator = Generator(
            img_size, latent_size, map_layers, channel_multiplier=channel_multiplier
        ).to(device)

        checkpoint = torch.load(checkpoint_path, map_location=device)

        self.generator.load_state_dict(checkpoint["g_ema"], strict=True)

        with torch.no_grad():
            self.mean_latent = self.generator.mean_latent(4096)

    def get_all_layers(self):
        return list(self.generator.children())

    def get_training_layers(self, phase):

        if phase == 'texture':
            # learned constant + first convolution + layers 3-10
            return list(self.get_all_layers())[1:3] + list(self.get_all_layers()[4][2:10])   
        if phase == 'shape':
            # layers 1-2
             return list(self.get_all_layers())[1:3] + list(self.get_all_layers()[4][0:2])
        if phase == 'no_fine':
            # const + layers 1-10
             return list(self.get_all_layers())[1:3] + list(self.get_all_layers()[4][:10])
        if phase == 'shape_expanded':
            # const + layers 1-10
             return list(self.get_all_layers())[1:3] + list(self.get_all_layers()[4][0:3])
        if phase == 'all':
            # everything, including mapping and ToRGB
            return self.get_all_layers() 
        else: 
            # everything except mapping and ToRGB
            return list(self.get_all_layers())[1:3] + list(self.get_all_layers()[4][:])  

    def freeze_layers(self, layer_list=None):
        '''
        Disable training for all layers in list.
        '''
        if layer_list is None:
            self.freeze_layers(self.get_all_layers())
        else:
            for layer in layer_list:
                requires_grad(layer, False)

    def unfreeze_layers(self, layer_list=None):
        '''
        Enable training for all layers in list.
        '''
        if layer_list is None:
            self.unfreeze_layers(self.get_all_layers())
        else:
            for layer in layer_list:
                requires_grad(layer, True)

    def style(self, styles):
        '''
        Convert z codes to w codes.
        '''
        styles = [self.generator.style(s) for s in styles]
        return styles

    def get_s_code(self, styles, input_is_latent=False):
        return self.generator.get_s_code(styles, input_is_latent)

    def modulation_layers(self):
        return self.generator.modulation_layers

    #TODO Maybe convert to kwargs
    def forward(self,
        styles,
        return_latents=False,
        inject_index=None,
        truncation=1,
        truncation_latent=None,
        input_is_latent=False,
        input_is_s_code=False,
        noise=None,
        randomize_noise=True):
        return self.generator(styles, return_latents=return_latents, truncation=truncation, truncation_latent=self.mean_latent, noise=noise, randomize_noise=randomize_noise, input_is_latent=input_is_latent, input_is_s_code=input_is_s_code)

class SG2Discriminator(torch.nn.Module):
    def __init__(self, checkpoint_path, img_size=256, channel_multiplier=2, device='cuda:0'):
        super(SG2Discriminator, self).__init__()

        self.discriminator = Discriminator(
            img_size, channel_multiplier=channel_multiplier
        ).to(device)

        checkpoint = torch.load(checkpoint_path, map_location=device)

        self.discriminator.load_state_dict(checkpoint["d"], strict=True)

    def get_all_layers(self):
        return list(self.discriminator.children())

    def get_training_layers(self):
        return self.get_all_layers() 

    def freeze_layers(self, layer_list=None):
        '''
        Disable training for all layers in list.
        '''
        if layer_list is None:
            self.freeze_layers(self.get_all_layers())
        else:
            for layer in layer_list:
                requires_grad(layer, False)

    def unfreeze_layers(self, layer_list=None):
        '''
        Enable training for all layers in list.
        '''
        if layer_list is None:
            self.unfreeze_layers(self.get_all_layers())
        else:
            for layer in layer_list:
                requires_grad(layer, True)

    def forward(self, images):
        return self.discriminator(images)

class ZSSGAN(pl.LightningModule):
    def __init__(self, args):
        super(ZSSGAN, self).__init__()

        self.args = args

#         self.device = 'cuda:0'

        # Set up frozen (source) generator
        self.generator_frozen = SG2Generator(args.frozen_gen_ckpt, img_size=args.size, channel_multiplier=args.channel_multiplier)#.to(self.device)
        self.generator_frozen.freeze_layers()
        self.generator_frozen.eval()

        # Set up trainable (target) generator
        self.generator_trainable = SG2Generator(args.train_gen_ckpt, img_size=args.size, channel_multiplier=args.channel_multiplier)#.to(self.device)
        self.generator_trainable.freeze_layers()
        self.generator_trainable.unfreeze_layers(self.generator_trainable.get_training_layers(args.phase))
        self.generator_trainable.train()

        # Losses
        self.clip_loss_models = {model_name: CLIPLoss('cuda',#self.device, 
                                                      lambda_direction=args.lambda_direction, 
                                                      lambda_patch=args.lambda_patch, 
                                                      lambda_global=args.lambda_global, 
                                                      lambda_manifold=args.lambda_manifold, 
                                                      lambda_texture=args.lambda_texture,
                                                      clip_model=model_name) 
                                for model_name in args.clip_models}

        self.clip_model_weights = {model_name: weight for model_name, weight in zip(args.clip_models, args.clip_model_weights)}

        self.mse_loss  = torch.nn.MSELoss()
        self.g_reg_ratio = args.g_reg_ratio # for Adam optimizer
        self.lr = args.lr
        
        self.source_model_type = args.source_model_type # model weights type: ffhq, cat, dog
        self.source_class = args.source_class #eg. photo
        self.target_class = args.target_class #eg. sketch

        self.auto_layer_k     = args.auto_layer_k
        self.auto_layer_iters = args.auto_layer_iters
        
        if args.target_img_list is not None:
            self.set_img2img_direction()
        
        # image seed for training
        self.n_sample = args.n_sample
        self.init_fixed_z(self.n_sample)
        self.sample_truncation = args.sample_truncation
        self.batch = args.batch
        self.mixing = args.mixing
        
    def init_fixed_z(self, n_sample):
        self.fixed_z = torch.randn(n_sample, 512, 
#                                    device=self.device
                                  )
        return None
    
    def configure_optimizers(self):
        g_reg_ratio = self.g_reg_ratio
        optim = torch.optim.Adam(
                    self.generator_trainable.parameters(),
                    lr=self.lr * g_reg_ratio,
                    betas=(0 ** g_reg_ratio, 0.99 ** g_reg_ratio),
                    )
        return optim
    
    def set_img2img_direction(self):
        with torch.no_grad():
            sample_z  = torch.randn(self.args.img2img_batch, 512, 
#                                     device=self.device
                                   )
            generated = self.generator_trainable([sample_z])[0]

            for _, model in self.clip_loss_models.items():
                direction = model.compute_img2img_direction(generated, self.args.target_img_list)

                model.target_direction = direction

    def determine_opt_layers(self):

        sample_z = torch.randn(self.args.auto_layer_batch, 512, 
#                                device=self.device
                              )

        initial_w_codes = self.generator_frozen.style([sample_z])
        initial_w_codes = initial_w_codes[0].unsqueeze(1).repeat(1, self.generator_frozen.generator.n_latent, 1)

        w_codes = torch.Tensor(initial_w_codes.cpu().detach().numpy())#.to(self.device)

        w_codes.requires_grad = True

        w_optim = torch.optim.SGD([w_codes], lr=0.01)

        for _ in range(self.auto_layer_iters):
            w_codes_for_gen = w_codes.unsqueeze(0)
            generated_from_w = self.generator_trainable(w_codes_for_gen, input_is_latent=True)[0]

            w_loss = [self.clip_model_weights[model_name] * self.clip_loss_models[model_name].global_clip_loss(generated_from_w, self.target_class) for model_name in self.clip_model_weights.keys()]
            w_loss = torch.sum(torch.stack(w_loss))
            
            w_optim.zero_grad()
            w_loss.backward()
            w_optim.step()
        
        layer_weights = torch.abs(w_codes - initial_w_codes).mean(dim=-1).mean(dim=0)
        chosen_layer_idx = torch.topk(layer_weights, self.auto_layer_k)[1].cpu().numpy()

        all_layers = list(self.generator_trainable.get_all_layers())

        conv_layers = list(all_layers[4])
        rgb_layers = list(all_layers[6]) # currently not optimized

        idx_to_layer = all_layers[2:4] + conv_layers # add initial convs to optimization

        chosen_layers = [idx_to_layer[idx] for idx in chosen_layer_idx] 

        # uncomment to add RGB layers to optimization.

        # for idx in chosen_layer_idx:
        #     if idx % 2 == 1 and idx >= 3 and idx < 14:
        #         chosen_layers.append(rgb_layers[(idx - 3) // 2])

        # uncomment to add learned constant to optimization
        # chosen_layers.append(all_layers[1])
                
        return chosen_layers

    def forward(
        self,
        styles,
        return_latents=False,
        inject_index=None,
        truncation=1,
        truncation_latent=None,
        input_is_latent=False,
        noise=None,
        randomize_noise=True,
    ):

        if self.training and self.auto_layer_iters > 0:
            self.generator_trainable.unfreeze_layers()
            train_layers = self.determine_opt_layers()

            if not isinstance(train_layers, list):
                train_layers = [train_layers]

            self.generator_trainable.freeze_layers()
            self.generator_trainable.unfreeze_layers(train_layers)

        with torch.no_grad():
            if input_is_latent:
                w_styles = styles
            else:
                w_styles = self.generator_frozen.style(styles)
            
            frozen_img = self.generator_frozen(w_styles, input_is_latent=True, truncation=truncation, randomize_noise=randomize_noise)[0]

        trainable_img = self.generator_trainable(w_styles, input_is_latent=True, truncation=truncation, randomize_noise=randomize_noise)[0]
        
        clip_loss = torch.sum(torch.stack([self.clip_model_weights[model_name] * self.clip_loss_models[model_name](frozen_img, self.source_class, trainable_img, self.target_class) for model_name in self.clip_model_weights.keys()]))

        return [frozen_img, trainable_img], clip_loss
    
    def training_step(self, batch, batch_idx, optimizer_idx = 0):
        sample_z, _ = batch
        [sampled_src, sampled_dst], clip_loss = self(sample_z)
        
        return clip_loss
    
    def generate_samples_jupyter(self, sample_dir, use_fixed_z=False, n_sample=9, 
                                 prefix_str="sample", postfix_num=0, frame_size=(1024, 256)):
        self.eval()
        with torch.no_grad():
            if use_fixed_z:
                samples = self.fixed_z.to('cuda')
            else:
                samples = torch.randn(n_sample, 512, device='cuda')
                
            [sampled_src, sampled_dst], loss = self([samples], truncation=self.sample_truncation)

            if self.source_model_type == 'car':
                sampled_dst = sampled_dst[:, :, 64:448, :]

            grid_rows = 4
            save_images(sampled_dst, sample_dir, prefix_str, grid_rows, postfix_num)
            img = Image.open(os.path.join(sample_dir, f"{prefix_str}_{str(postfix_num).zfill(6)}.jpg")).resize(frame_size)
            display(img)
        
    def pivot(self):
        par_frozen = dict(self.generator_frozen.named_parameters())
        par_train  = dict(self.generator_trainable.named_parameters())

        for k in par_frozen.keys():
            par_frozen[k] = par_train[k]
