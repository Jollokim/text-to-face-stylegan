import torch
import torch.nn as nn

from utils import get_w, get_noise
from tqdm import tqdm

from torch.utils.tensorboard import SummaryWriter




def generate_fakes(batch_size: int, generator, map_net, n_blocks, w_dim, device,):
    """
    ### Generate images
    This generate images using the generator
    """

    # Get $w$
    w = get_w(n_blocks, batch_size, map_net, w_dim, device)
    # Get noise
    noise = get_noise(n_blocks, batch_size, device)

    # Generate images
    images = generator(w, noise)

    # Return images and $w$
    return images, w


step_counter = 0

def train_one_epoch(gan: dict[str, nn.Module],
                    gan_criterion: dict[str, nn.Module],
                    gan_opt: dict[str, nn.Module],
                    real_dataloader: torch.utils.data.DataLoader,
                    device: torch.device,
                    n_blocks: int,
                    w_dim: int,
                    gradient_penalty,
                    path_length_penalty,
                    gradient_penalty_coefficient: float = 10.,
                    lazy_gradient_penalty_interval: int=4, 
                    lazy_path_penalty_interval: int = 32,
                    writer: SummaryWriter=None
                    ):
    global step_counter
    

    for comp in gan.keys():
        gan[comp].train(True)

    discriminator = gan['discriminator']
    generator = gan['generator']
    map_net = gan['map_net']

    discriminator_loss = gan_criterion['discriminator']
    generator_loss = gan_criterion['generator']
    

    discriminator_optimizer = gan_opt['discriminator']
    generator_optimizer = gan_opt['generator']
    map_net_optimizer = gan_opt['map_net']

    for real in tqdm(real_dataloader):
        discriminator_optimizer.zero_grad()

        # step discriminator
        real = real.to(device, non_blocking=True)

        if (step_counter % lazy_gradient_penalty_interval) == 0:
            real.requires_grad_()

        fake, w = generate_fakes(real.shape[0], generator, map_net, n_blocks, w_dim, device)

        real_out = discriminator(real)

        fake_out = discriminator(fake.detach())

        real_loss, fake_loss = discriminator_loss(real_out, fake_out)
        disc_loss = real_loss + fake_loss

        writer.add_scalar('disc real_loss', real_loss, step_counter)
        writer.add_scalar('disc fake_loss', fake_loss, step_counter)

        if (step_counter % lazy_gradient_penalty_interval) == 0:
            # Calculate and log gradient penalty
            gp = gradient_penalty(real, real_out)
            disc_loss + 0.5 * gradient_penalty_coefficient * gp * lazy_gradient_penalty_interval

        writer.add_scalar('disc loss', disc_loss, step_counter)

        # print('disc loss', disc_loss, real_loss, fake_loss)

        disc_loss.backward()

        # Clip gradients for stabilization
        torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=1.0)

        discriminator_optimizer.step()

        # step generator
        generator_optimizer.zero_grad()
        map_net_optimizer.zero_grad()

        fake, w = generate_fakes(real.shape[0], generator, map_net, n_blocks, w_dim, device)

        fake_out = discriminator(fake)

        gen_loss = generator_loss(fake_out)

        # Add path length penalty
        if (step_counter % lazy_path_penalty_interval) == 0:
            # Calculate path length penalty
            plp = path_length_penalty(w, fake)
            # Ignore if `nan`
            if not torch.isnan(plp):
                gen_loss = gen_loss + plp

        correct_n = 0
        for out in fake_out:
            if out[0] < 0:
                correct_n += 1

        writer.add_scalar('fakes accuracy on discriminator', correct_n/fake_out.shape[0], step_counter)

        writer.add_scalar('gen loss', gen_loss, step_counter)

        # print('gen loss', gen_loss)

        gen_loss.backward()

        # torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=1.0)
        # torch.nn.utils.clip_grad_norm_(map_net.parameters(), max_norm=1.0)

        generator_optimizer.step()
        map_net_optimizer.step()

        step_counter += 1



        



def step():
    pass
