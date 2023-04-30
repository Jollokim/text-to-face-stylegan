from stylegan2 import Generator, Discriminator, MappingNetwork
import math
import torch

from utils import get_noise
from losses import DiscriminatorLoss, GeneratorLoss


def main():
    log_resolution = int(math.log2(512))
    
    # make divisible by for 4 for simplicity
    batch_size = 4

    w_dim = 512
    n_blocks = 9

    generator = Generator(log_resolution, w_dim)
    discriminator = Discriminator(log_resolution)
    map_net = MappingNetwork(w_dim, 8)

    # Discriminator and generator losses
    discriminator_loss = DiscriminatorLoss()
    generator_loss = GeneratorLoss()

    z = torch.randn(batch_size, w_dim)
    noise = get_noise(n_blocks, batch_size)

    print('z.shape', z.shape)

    w = map_net(z)
    print('w.shape', w.shape)
    w = w[None, :, :].expand(n_blocks, -1, -1)

    print('transformed w.shape', w.shape)

    print(noise[1][0].shape)
    print(noise[0][1].shape)

    out_img = generator(w, noise)
    real_img = torch.randn((4, 3, 512, 512))

    print(out_img.shape)

    out_dis_fake = discriminator(out_img.detach())
    out_dis_real = discriminator(real_img)

    print(out_dis_fake.shape)

    real_loss, fake_loss = discriminator_loss(out_dis_real, out_dis_fake)
    disc_loss = real_loss + fake_loss

    print(disc_loss)

    gen_loss = generator_loss()

    






    




if __name__ == '__main__':
    main()