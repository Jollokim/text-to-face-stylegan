import clip
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms.functional as F

from utils import get_noise


def generate_image_from_description(description: str, generator, clip_model: clip.model.CLIP, device):
    """
    ### Generate images
    This generate images using the generator
    """

    n_blocks = generator.n_blocks

    # Get $w$
    # w = get_w(n_blocks, batch_size, map_net, w_dim, device)
    descriptions_tok = clip.tokenize([description]).to(device)
    encoded_text = clip_model.encode_text(descriptions_tok)

    encoded_text = encoded_text[None, :, :].expand(n_blocks, -1, -1).float()

    # Get noise
    noise = get_noise(n_blocks, 1, device)

    # Generate images
    images = generator(encoded_text, noise)

    # Return images and $w$
    return images



plt.rcParams["savefig.bbox"] = 'tight'


def show(imgs):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fig, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = F.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    fig.show()