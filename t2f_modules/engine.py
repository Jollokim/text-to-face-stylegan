import clip
import torch
import torch.nn as nn
import torchvision
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from utils import get_noise, get_w


def generate_fakes(descriptions:list[str], batch_size: int, generator, clip_model: clip.model.CLIP, n_blocks, w_dim, device):
    """
    ### Generate images
    This generate images using the generator
    """

    # Get $w$
    # w = get_w(n_blocks, batch_size, map_net, w_dim, device)
    descriptions_tok = clip.tokenize(descriptions).to(device)
    encoded_text = clip_model.encode_text(descriptions_tok)

    encoded_text = encoded_text[None, :, :].expand(n_blocks, -1, -1).float()

    # Get noise
    noise = get_noise(n_blocks, batch_size, device)

    # Generate images
    images = generator(encoded_text, noise)

    # Return images and $w$
    return images, encoded_text


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
                    clip_model: clip.model.CLIP,
                    clip_preprocess: torchvision.transforms.transforms.Compose,
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

    discriminator_loss = gan_criterion['discriminator']
    generator_loss = gan_criterion['generator']
    clip_crit = gan_criterion['clip_loss']
    

    discriminator_optimizer = gan_opt['discriminator']
    generator_optimizer = gan_opt['generator']

    for image, image_transformed, descriptions in tqdm(real_dataloader):
        # descriptions_tok = clip.tokenize(descriptions).to(device)
        # encoded_text = clip_model.encode_text(descriptions_tok) # some issue with the encoded text when calculating pathlength

        # ######### step discriminator #########
        discriminator_optimizer.zero_grad()

        image_transformed = image_transformed.to(device, non_blocking=True)

        if (step_counter % lazy_gradient_penalty_interval) == 0:
            image_transformed.requires_grad_()

        fake, enc = generate_fakes(descriptions, image_transformed.shape[0], generator, clip_model, n_blocks, w_dim, device)

        real_out = discriminator(image_transformed)

        fake_out = discriminator(fake.detach())

        real_loss, fake_loss = discriminator_loss(real_out, fake_out)
        disc_loss = real_loss + fake_loss

        writer.add_scalar('disc real_loss', real_loss, step_counter)
        writer.add_scalar('disc fake_loss', fake_loss, step_counter)

        if (step_counter % lazy_gradient_penalty_interval) == 0:
            # Calculate and log gradient penalty
            gp = gradient_penalty(image_transformed, real_out)
            disc_loss + 0.5 * gradient_penalty_coefficient * gp * lazy_gradient_penalty_interval

        writer.add_scalar('disc loss', disc_loss, step_counter)

        # print('disc loss', disc_loss, real_loss, fake_loss)

        disc_loss.backward()

        # Clip gradients for stabilization
        torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=1.0)

        discriminator_optimizer.step()

        # #########  step generator #########  
        generator_optimizer.zero_grad()

        fake, enc = generate_fakes(descriptions, image_transformed.shape[0], generator, clip_model, n_blocks, w_dim, device)

        fake_out = discriminator(fake)

        gen_loss = generator_loss(fake_out)

        clip_loss = clip_crit(descriptions, fake)

        gen_loss = gen_loss + clip_loss

        # Add path length penalty
        if (step_counter % lazy_path_penalty_interval) == 0:
            # Calculate path length penalty
            plp = path_length_penalty(enc, fake)
            # Ignore if `nan`
            if not torch.isnan(plp):
                gen_loss = gen_loss + plp

        correct_n = 0
        for out in fake_out:
            if out[0] < 0:
                correct_n += 1


        writer.add_scalar('fakes accuracy on discriminator', correct_n/fake_out.shape[0], step_counter)

        writer.add_scalar('gen cosine loss', clip_loss, step_counter)
        writer.add_scalar('gen loss', gen_loss, step_counter)

        # print('gen loss', gen_loss)

        gen_loss.backward()

        # torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=1.0)

        generator_optimizer.step()

        step_counter += 1

