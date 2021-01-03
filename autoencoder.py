import torch
import torch.nn as nn
import torch.nn.functional as F


class EncoderCNN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        modules = []

        # TODO:
        #  Implement a CNN. Save the layers in the modules list.
        #  The input shape is an image batch: (N, in_channels, H_in, W_in).
        #  The output shape should be (N, out_channels, H_out, W_out).
        #  You can assume H_in, W_in >= 64.
        #  Architecture is up to you, but it's recommended to use at
        #  least 3 conv layers. You can use any Conv layer parameters,
        #  use pooling or only strides, use any activation functions,
        #  use BN or Dropout, etc.
        # ====== YOUR CODE: ======
        # Model architecture
        channels = [in_channels, 64, 128, 256, out_channels]
        kernel_size = (5, 5)
        padding = 0
        dilation = 2
        batch_norm = True
        dropout = 0.2

        # Create the layers
        for channel_idx in range(len(channels)-1):
            in_features, out_features = channels[channel_idx], channels[channel_idx+1]
            layer = nn.Conv2d(in_channels=in_features, out_channels=out_features,
                              kernel_size=kernel_size, padding=padding, dilation=dilation)
            modules.append(layer)

            if batch_norm:
                batch_norm_layer = nn.BatchNorm2d(num_features=out_features)
                modules.append(batch_norm_layer)

            if dropout > 0:
                dropout_layer = nn.Dropout2d(p=dropout)
                modules.append(dropout_layer)
        # ========================
        self.cnn = nn.Sequential(*modules)

    def forward(self, x):
        return self.cnn(x)


class DecoderCNN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        modules = []

        # TODO:
        #  Implement the "mirror" CNN of the encoder.
        #  For example, instead of Conv layers use transposed convolutions,
        #  instead of pooling do unpooling (if relevant) and so on.
        #  The architecture does not have to exactly mirror the encoder
        #  (although you can), however the important thing is that the
        #  output should be a batch of images, with same dimensions as the
        #  inputs to the Encoder were.
        # ====== YOUR CODE: ======
        # Model architecture
        channels = [in_channels, 256, 128, 64, out_channels]
        kernel_size = (5, 5)
        padding = 0
        dilation = 2
        batch_norm = True
        dropout = 0.2
        stride = 1

        # Create the layers
        for channel_idx in range(len(channels)-1):
            in_features, out_features = channels[channel_idx], channels[channel_idx+1]
            layer = nn.ConvTranspose2d(in_channels=in_features, out_channels=out_features,
                                       kernel_size=kernel_size, padding=padding, dilation=dilation, stride=stride)
            modules.append(layer)

            if batch_norm:
                batch_norm_layer = nn.BatchNorm2d(num_features=out_features)
                modules.append(batch_norm_layer)

            if dropout > 0:
                dropout_layer = nn.Dropout2d(p=dropout)
                modules.append(dropout_layer)
        # ========================
        self.cnn = nn.Sequential(*modules)

    def forward(self, h):
        # Tanh to scale to [-1, 1] (same dynamic range as original images).
        return torch.tanh(self.cnn(h))


class VAE(nn.Module):
    def __init__(self, features_encoder, features_decoder, in_size, z_dim):
        """
        :param features_encoder: Instance of an encoder the extracts features
        from an input.
        :param features_decoder: Instance of a decoder that reconstructs an
        input from it's features.
        :param in_size: The size of one input (without batch dimension).
        :param z_dim: The latent space dimension.
        """
        super().__init__()
        self.features_encoder = features_encoder
        self.features_decoder = features_decoder
        self.z_dim = z_dim

        self.features_shape, n_features = self._check_features(in_size)

        # TODO: Add more layers as needed for encode() and decode().
        # ====== YOUR CODE: ======
        # Add linear layers for encoder (mu and sigma)
        self.layer_w_mu = nn.Linear(in_features=n_features, out_features=self.z_dim, bias=True)
        self.layer_w_sigma2 = nn.Linear(in_features=n_features, out_features=self.z_dim)
        self.flatten_layer = nn.Flatten()
        # Add layers for the encoder
        self.z_hbar = nn.Linear(in_features=self.z_dim, out_features=n_features)
        # self.unflatten_layer = nn.Unflatten(dim=0, unflattened_size=self.features_shape)  # needed pytorch = 1.7.0
        # ========================

    def _check_features(self, in_size):
        device = next(self.parameters()).device
        with torch.no_grad():
            # Make sure encoder and decoder are compatible
            x = torch.randn(1, *in_size, device=device)
            h = self.features_encoder(x)
            xr = self.features_decoder(h)
            assert xr.shape == x.shape
            # Return the shape and number of encoded features
            return h.shape[1:], torch.numel(h) // h.shape[0]

    def encode(self, x):
        # TODO:
        #  Sample a latent vector z given an input x from the posterior q(Z|x).
        #  1. Use the features extracted from the input to obtain mu and
        #     log_sigma2 (mean and log variance) of q(Z|x).
        #  2. Apply the reparametrization trick to obtain z.
        # ====== YOUR CODE: ======
        # Add a flattening layer
        h = self.flatten_layer(self.features_encoder(x))
        mu = self.layer_w_mu(h)
        log_sigma2 = self.layer_w_sigma2(h)

        # Sample u from normal distribution N(0,1)
        u = torch.randn(2)
        z = mu + u * torch.exp(0.5 * log_sigma2)
        # ========================

        return z, mu, log_sigma2

    def decode(self, z):
        # TODO:
        #  Convert a latent vector back into a reconstructed input.
        #  1. Convert latent z to features h with a linear layer.
        #  2. Apply features decoder.
        # ====== YOUR CODE: ======
        h_bar = self.z_hbar(z)
        # h_bar = self.unflatten_layer(h_bar) # needed pytorch = 1.7.0 for that
        h_bar = torch.reshape(h_bar, [-1, *self.features_shape])
        x_rec = self.features_decoder(h_bar)
        # ========================

        # Scale to [-1, 1] (same dynamic range as original images).
        return torch.tanh(x_rec)

    def sample(self, n):
        samples = []
        device = next(self.parameters()).device
        with torch.no_grad():
            # TODO:
            #  Sample from the model. Generate n latent space samples and
            #  return their reconstructions.
            #  Notes:
            #  - Remember that this means using the model for INFERENCE.
            #  - We'll ignore the sigma2 parameter here:
            #    Instead of sampling from N(psi(z), sigma2 I), we'll just take
            #    the mean, i.e. psi(z).
            # ====== YOUR CODE: ======
            z_shape = [1, self.z_dim]
            # Loop over number of samples needed
            for sample_idx in range(n):
                z = torch.randn(*z_shape)
                x_sample = self.decode(z)
                samples.append(x_sample)
            samples = torch.cat(samples, dim=0)
            # ========================

        # Detach and move to CPU for display purposes
        samples = [s.detach().cpu() for s in samples]
        return samples

    def forward(self, x):
        z, mu, log_sigma2 = self.encode(x)
        return self.decode(z), mu, log_sigma2


def vae_loss(x, xr, z_mu, z_log_sigma2, x_sigma2):
    """
    Point-wise loss function of a VAE with latent space of dimension z_dim.
    :param x: Input image batch of shape (N,C,H,W).
    :param xr: Reconstructed (output) image batch.
    :param z_mu: Posterior mean (batch) of shape (N, z_dim).
    :param z_log_sigma2: Posterior log-variance (batch) of shape (N, z_dim).
    :param x_sigma2: Likelihood variance (scalar).
    :return:
        - The VAE loss
        - The data loss term
        - The KL divergence loss term
    all three are scalars, averaged over the batch dimension.
    """
    loss, data_loss, kldiv_loss = None, [], []
    # TODO:
    #  Implement the VAE pointwise loss calculation.
    #  Remember:
    #  1. The covariance matrix of the posterior is diagonal.
    #  2. You need to average over the batch dimension.
    # ====== YOUR CODE: ======
    n_batches = x.shape[0]
    dx = torch.numel(x) / n_batches
    dz = z_mu.shape[1]

    x = torch.reshape(x, [n_batches, -1])  # size [N, dx]
    xr = torch.reshape(xr, [n_batches, -1])  # size [N, dx]
    z_sigma2 = torch.exp(z_log_sigma2)  # size [N, dx]
    trace = torch.sum(z_sigma2, dim=1)  # size [N, dx]
    det = torch.prod(z_sigma2, dim=1)  # size [N, dx]

    data_loss = torch.mean(torch.sum((x - xr) ** 2, dim=1) / (x_sigma2 * dx))
    kldiv_loss = torch.mean(trace + torch.sum(z_mu ** 2, dim=1) - dz - torch.log(det))
    loss = data_loss + kldiv_loss
    # ========================

    return loss, data_loss, kldiv_loss
