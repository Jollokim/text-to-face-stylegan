import argparse
import math

import torch
import torch.nn as nn

from simple_modules.dataset import SimpleDataset
import stylegan2
from simple_modules.engine import train_one_epoch
from losses import GeneratorLoss, DiscriminatorLoss

from torch.utils.tensorboard import SummaryWriter
import torchvision.utils as vutils
from simple_modules.engine import generate_fakes

import os




def get_args_parser():
    parser = argparse.ArgumentParser('train', add_help=False)

    # Data
    parser.add_argument('--real_folder', type=str, default='data/images',
                        help='folder for real images')
    
    # Model
    parser.add_argument('--w_dim', type=int, default=512,
                        help='')
    parser.add_argument('--image_size', type=int, default=64,
                        help='')
    parser.add_argument('--gen_lr', type=float, default=1e-3,
                        help='')
    parser.add_argument('--map_lr', type=float, default=1e-5,
                        help='batch size')
    parser.add_argument('--disc_lr', type=float, default=1e-3,
                        help='')
    
    parser.add_argument('--weight_dir', type=str, default=None,
                        help='')


    # training
    parser.add_argument('--batch_size', type=int, default=64,
                        help='')
    parser.add_argument('--epochs', type=int, default=300,
                        help='')
    
    # logging
    parser.add_argument('--name', type=str, default='test',
                        help='')
    parser.add_argument('--overwrite', action='store_true',
                        help='')
    parser.add_argument('--save_every', type=int, default=100,
                        help='')
    
    


    return parser


def main(args):
    
    # weight and logs
    log_path = f'runs/{args.name}'
    os.makedirs(log_path, exist_ok=args.overwrite)

    # model related
    w_dim = args.w_dim
    image_size = args.image_size
    log_resolution = int(math.log2(image_size))

    gen_lr = args.gen_lr
    map_lr = args.map_lr
    disc_lr = args.disc_lr

    adam_betas = (0.0, 0.99)

    # generator related
    generator = stylegan2.Generator(log_resolution, w_dim)
    map_net = stylegan2.MappingNetwork(w_dim, 8)

    print('Generator')
    print(generator)

    n_blocks = generator.n_blocks

    print('Number of generator blocks:', n_blocks)

    generator_optimizer = torch.optim.Adam(
            generator.parameters(),
            lr=gen_lr, betas=adam_betas
        )
    map_net_optimizer = torch.optim.Adam(
        map_net.parameters(),
        lr=map_lr, betas=adam_betas
        
    )

    generator_loss = GeneratorLoss()
    path_length_penalty = stylegan2.PathLengthPenalty(0.99)

    # discriminator related
    discriminator = stylegan2.Discriminator(log_resolution)
    discriminator_optimizer = torch.optim.Adam(
        discriminator.parameters(),
        lr=disc_lr, betas=adam_betas
    )

    print('Discriminator')
    print(discriminator)

    discriminator_loss = DiscriminatorLoss()
    gradient_penalty = stylegan2.GradientPenalty()
    

    

    # do not train on cpu
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"

        # The Dataparallel does something strange the generator doesn't like, prob because of all the manual handling of vectors
        # if torch.cuda.device_count() > 1:
            # generator = nn.DataParallel(generator)
            # map_net = nn.DataParallel(map_net)
            # discriminator = nn.DataParallel(discriminator)
    
    generator.to(device)
    map_net.to(device)
    discriminator.to(device)

    if args.weight_dir is not None:
        # generator.load_state_dict(torch.load(f'{args.weight_dir}/generator.pt'))
        generator.load_state_dict(torch.load(f'{args.weight_dir}/generator.pt'))
        map_net.load_state_dict(torch.load(f'{args.weight_dir}/map_net.pt'))
        discriminator.load_state_dict(torch.load(f'{args.weight_dir}/discriminator.pt'))
        print('Successfully loaded weights!')

    generator_loss.to(device)
    path_length_penalty.to(device)

    discriminator_loss.to(device)

    gan = {'generator': generator, 'map_net': map_net, 'discriminator': discriminator}
    gan_opt = {'generator': generator_optimizer, 'map_net': map_net_optimizer, 'discriminator': discriminator_optimizer}
    gan_crit = {'generator': generator_loss, 'discriminator': discriminator_loss}

    dataset = SimpleDataset(args.real_folder, image_size=image_size )
    dataloader = data_loader_train = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=5,
        shuffle=True,
        drop_last=True, 
        pin_memory=True
    )

    writer = SummaryWriter(f'{log_path}/tensorboard', filename_suffix=args.name)
    
    for epoch in range(args.epochs):
        with torch.no_grad():
                fake, _ = generate_fakes(16, generator, map_net, n_blocks, w_dim, device)
                grid = vutils.make_grid(fake, padding=2, normalize=True)
                writer.add_image('fakes generated at epoch', grid, epoch)

        print('epoch:', epoch)
        train_one_epoch(gan, gan_crit, gan_opt, dataloader, device, n_blocks, w_dim, gradient_penalty, path_length_penalty, writer=writer)

        if (epoch % args.save_every) == 0:
            torch.save(generator.state_dict(), f'{log_path}/checkpoint_generator.pt')
            torch.save(map_net.state_dict(), f'{log_path}/checkpoint_map_net.pt')
            torch.save(discriminator.state_dict(), f'{log_path}/checkpoint_discriminator.pt')

    torch.save(generator.state_dict(), f'{log_path}/generator.pt')
    torch.save(map_net.state_dict(), f'{log_path}/map_net.pt')
    torch.save(discriminator.state_dict(), f'{log_path}/discriminator.pt')


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser('train ', parents=[get_args_parser()])
    args = arg_parser.parse_args()

    main(args)
