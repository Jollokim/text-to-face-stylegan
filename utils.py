import torch


def get_w(n_blocks, batch_size: int, map_net, w_dim, device):
        """
        ### Sample $w$
        This samples $z$ randomly and get $w$ from the mapping network.
        We also apply style mixing sometimes where we generate two latent variables
        $z_1$ and $z_2$ and get corresponding $w_1$ and $w_2$.
        Then we randomly sample a cross-over point and apply $w_1$ to
        the generator blocks before the cross-over point and
        $w_2$ to the blocks after.
        """

        # Mix styles
        # if torch.rand(()).item() < self.style_mixing_prob:
        #     # Random cross-over point
        #     cross_over_point = int(torch.rand(()).item() * self.n_gen_blocks)
        #     # Sample $z_1$ and $z_2$
        #     z2 = torch.randn(batch_size, self.d_latent).to(self.device)
        #     z1 = torch.randn(batch_size, self.d_latent).to(self.device)
        #     # Get $w_1$ and $w_2$
        #     w1 = self.mapping_network(z1)
        #     w2 = self.mapping_network(z2)
        #     # Expand $w_1$ and $w_2$ for the generator blocks and concatenate
        #     w1 = w1[None, :, :].expand(cross_over_point, -1, -1)
        #     w2 = w2[None, :, :].expand(self.n_gen_blocks - cross_over_point, -1, -1)
        #     return torch.cat((w1, w2), dim=0)
        # Without mixing
        # else:
        # Sample $z$ and $z$
        z = torch.randn(batch_size, w_dim).to(device)
        # Get $w$ and $w$
        w = map_net(z)
        # Expand $w$ for the generator blocks
        return w[None, :, :].expand(n_blocks, -1, -1)

def get_noise(n_blocks, batch_size: int, device):
        """
        ### Generate noise
        This generates noise for each [generator block](index.html#generator_block)
        """
        # List to store noise
        noise = []
        # Noise resolution starts from $4$
        resolution = 4

        # Generate noise for each generator block
        for i in range(n_blocks):
            # The first block has only one $3 \times 3$ convolution
            if i == 0:
                n1 = None
            # Generate noise to add after the first convolution layer
            else:
                n1 = torch.randn(batch_size, 1, resolution, resolution, device=device)
            # Generate noise to add after the second convolution layer
            n2 = torch.randn(batch_size, 1, resolution, resolution, device=device)

            # Add noise tensors to the list
            noise.append((n1, n2))

            # Next block has $2 \times$ resolution
            resolution *= 2

        # Return noise tensors
        return noise