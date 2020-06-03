from argparse import ArgumentParser
from collections import OrderedDict
from typing import Tuple

import torchvision
from pytorch_lightning import LightningModule
import torch
import torch.nn.functional as F

from gan.discriminators import Discriminator
from gan.generators import Generator


class Gan(LightningModule):

    def __init__(
            self,
            img_shape: Tuple[int, int, int],
            latent_dim: int,
            lr: float = 1e-3,
            b1: float = 0.9,
            b2: float = 0.999,
            **kwargs
    ):

        super(Gan, self).__init__()

        self.latent_dim = latent_dim
        self.lr = lr
        self.b1 = b1
        self.b2 = b2

        # networks
        self.generator = Generator(latent_dim=latent_dim, img_shape=img_shape)
        self.discriminator = Discriminator(img_shape=img_shape)

        # cache for generated images
        self.generated_imgs = None
        self.last_imgs = None

    def forward(self, z):
        return self.generator(z)

    def adversarial_loss(self, y_hat, y):
        return F.binary_cross_entropy(y_hat, y)

    def training_step(self, batch, batch_nb, optimizer_idx):
        imgs, _ = batch
        self.last_imgs = imgs

        # train generator
        if optimizer_idx == 0:
            # sample noise
            z = torch.randn(imgs.shape[0], self.latent_dim)

            if self.on_gpu:
                z = z.cuda(imgs.device.index)

            self.generated_imgs = self(z)

            # ground truth result (ie: all fake)
            valid = torch.ones(imgs.size(0), 1)
            if self.on_gpu:
                valid = valid.cuda(imgs.device.index)

            # adversarial loss is binary cross-entropy
            g_loss = self.adversarial_loss(self.discriminator(self.generated_imgs), valid)
            tqdm_dict = {'g_loss': g_loss}
            output = OrderedDict({
                'loss': g_loss,
                'progress_bar': tqdm_dict,
                'log': tqdm_dict
            })
            return output

        # train discriminator
        if optimizer_idx == 1:
            valid = torch.ones(imgs.size(0), 1)
            if self.on_gpu:
                valid = valid.cuda(imgs.device.index)

            real_loss = self.adversarial_loss(self.discriminator(imgs), valid)

            fake = torch.zeros(imgs.size(0), 1)
            if self.on_gpu:
                fake = fake.cuda(imgs.device.index)

            fake_loss = self.adversarial_loss(
                self.discriminator(self.generated_imgs.detach()), fake)

            d_loss = (real_loss + fake_loss) / 2
            tqdm_dict = {'d_loss': d_loss}
            output = OrderedDict({
                'loss': d_loss,
                'progress_bar': tqdm_dict,
                'log': tqdm_dict
            })
            return output

    def configure_optimizers(self):
        lr = self.lr
        b1 = self.b1
        b2 = self.b2

        opt_g = torch.optim.Adam(self.generator.parameters(), lr=lr, betas=(b1, b2))
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=(b1, b2))
        return [opt_g, opt_d], []

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--latent_dim', type=int, default=128)
        parser.add_argument('--lr', type=float, default=1e-3)
        parser.add_argument('--b1', type=float, default=.9)
        parser.add_argument('--b2', type=float, default=.999)

        return parser

    def on_epoch_end(self):
        z = torch.randn(8, self.latent_dim)

        if self.on_gpu:
            z = z.cuda(self.last_imgs.device.index)

        sample_imgs = self(z)
        grid = torchvision.utils.make_grid(sample_imgs)
        self.logger.experiment.add_image(f'generated_images', grid, self.current_epoch)
