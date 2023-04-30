import torch.utils.data
import torchvision.transforms.functional as visionF
import torch.nn.functional as F
import clip


class DiscriminatorLoss(torch.nn.Module):
    """
    ## Discriminator Loss
    We want to find $w$ to maximize
    $$\mathbb{E}_{x \sim \mathbb{P}_r} [f_w(x)]- \mathbb{E}_{z \sim p(z)} [f_w(g_\theta(z))]$$,
    so we minimize,
    $$-\frac{1}{m} \sum_{i=1}^m f_w \big(x^{(i)} \big) +
     \frac{1}{m} \sum_{i=1}^m f_w \big( g_\theta(z^{(i)}) \big)$$
    """

    def forward(self, f_real: torch.Tensor, f_fake: torch.Tensor):
        """
        * `f_real` is $f_w(x)$
        * `f_fake` is $f_w(g_\theta(z))$
        This returns the a tuple with losses for $f_w(x)$ and $f_w(g_\theta(z))$,
        which are later added.
        They are kept separate for logging.
        """

        # We use ReLUs to clip the loss to keep $f \in [-1, +1]$ range.
        return F.relu(1 - f_real).mean(), F.relu(1 + f_fake).mean()


class GeneratorLoss(torch.nn.Module):
    """
    ## Generator Loss
    We want to find $\theta$ to minimize
    $$\mathbb{E}_{x \sim \mathbb{P}_r} [f_w(x)]- \mathbb{E}_{z \sim p(z)} [f_w(g_\theta(z))]$$
    The first component is independent of $\theta$,
    so we minimize,
    $$-\frac{1}{m} \sum_{i=1}^m f_w \big( g_\theta(z^{(i)}) \big)$$
    """

    def forward(self, f_fake: torch.Tensor):
        """
        * `f_fake` is $f_w(g_\theta(z))$
        """
        return -f_fake.mean()


class BaseCLIPLoss(torch.nn.Module):
    def __init__(self, clip_model, clip_preprocess, device) -> None:
        super().__init__()

        self.clip_model = clip_model
        self.clip_preprocess = clip_preprocess
        self.device = device

    def forward(self, descriptions, fake_imgs):
        text_tok = clip.tokenize(descriptions).to(self.device)
        text_enc = self.clip_model.encode_text(text_tok)

        fake_imgs_enc = []

        # very unfortunate that the CLIP model doesn't process images in batches...
        for i in range(fake_imgs.shape[0]):
            
            pil = visionF.to_pil_image(fake_imgs[i])
            img_preproc = self.clip_preprocess(pil).unsqueeze(0).to(self.device)

            fake_imgs_enc.append(self.clip_model.encode_image(img_preproc))

        fake_imgs_enc = torch.concat(fake_imgs_enc)

        return text_enc, fake_imgs_enc


class CosineClipLoss(BaseCLIPLoss):
    def __init__(self, clip_model, clip_preprocess, device) -> None:
        super().__init__(clip_model, clip_preprocess, device)

    def forward(self, descriptions, fake_imgs):
        text_enc, fake_imgs_enc = super().forward(descriptions, fake_imgs)

        sims = torch.cosine_similarity(text_enc, fake_imgs_enc)

        return -sims.mean()
    
class DotClipLoss(BaseCLIPLoss):
    def __init__(self, clip_model, clip_preprocess, device) -> None:
        super().__init__(clip_model, clip_preprocess, device)

    def forward(self, descriptions, fake_imgs):
        text_enc, fake_imgs_enc = super().forward(descriptions, fake_imgs)

        


if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model, preprocess = clip.load("ViT-B/32", device=device)

    loss_f = CosineClipLoss(model, preprocess, device)

    descriptions = ['first', 'second', 'third']
    images = torch.randn((3, 3, 64, 64))

    loss = loss_f(descriptions, images)

    print(loss)

