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
        nof1x1 = 100
        nof3x3 = 100
        nof2x2 = 200
        dropout = 0.3

        modules = [nn.Conv2d(in_channels=in_channels, out_channels=nof1x1, kernel_size=3, padding=2, stride=2),
                   nn.BatchNorm2d(nof1x1),
                   nn.LeakyReLU(),
                   nn.Conv2d(in_channels=nof1x1, out_channels=nof3x3, kernel_size=3, padding=2, stride=2),
                   nn.BatchNorm2d(nof3x3),
                   nn.LeakyReLU(),
                   nn.Conv2d(in_channels=nof3x3, out_channels=nof2x2, kernel_size=5, padding=2, stride=2),
                   nn.BatchNorm2d(nof2x2),
                   nn.LeakyReLU(),
                   nn.Dropout(dropout),
                   nn.Conv2d(nof2x2, out_channels, kernel_size=5, padding=2, stride=2),
                   nn.LeakyReLU()]

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

        nof1x1 = 100
        nof3x3 = 100
        nof2x2 = 200
        dropout = 0.3

        modules = [nn.ConvTranspose2d(in_channels=in_channels, out_channels=nof2x2, kernel_size=5, padding=2, stride=2),
                   nn.BatchNorm2d(nof2x2),
                   nn.ReLU(),
                   nn.ConvTranspose2d(in_channels=nof2x2, out_channels=nof3x3, kernel_size=5, padding=2, stride=2, output_padding=1),
                   nn.BatchNorm2d(nof3x3),
                   nn.ReLU(),
                   nn.ConvTranspose2d(in_channels=nof3x3, out_channels=nof1x1, kernel_size=3, padding=2, stride=2),
                   nn.BatchNorm2d(nof1x1),
                   nn.ReLU(),
                   nn.Dropout(dropout),
                   nn.ConvTranspose2d(in_channels=nof1x1, out_channels=out_channels, kernel_size=3, padding=2, stride=2, output_padding=1)
                   ]

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
        self.w_hmu = nn.Linear(n_features, z_dim)
        self.w_hsigma2 = nn.Linear(n_features, z_dim)
        self.w_hhat = nn.Linear(z_dim, n_features)
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

        # ====== YOUR CODE: ======
        device = next(self.parameters()).device
        x = x.to(device)

        # Sample a latent vector z given an input x from the posterior q(Z|x)
        h = self.features_encoder(x)
        h = h.view(h.size(0), -1)

        # Obtain mu and log_sigma2 (mean and log variance) of q(Z|x):
        mu = self.w_hmu(h)
        log_sigma2 = self.w_hsigma2(h)

        # Apply the reparametrization trick to obtain z:
        sigma_x = torch.sqrt(torch.exp(log_sigma2))
        z = mu + torch.randn_like(sigma_x) * sigma_x
        # ========================

        return z, mu, log_sigma2

    def decode(self, z):
        # TODO:
        #  Convert a latent vector back into a reconstructed input.
        #  1. Convert latent z to features h with a linear layer.
        #  2. Apply features decoder.
        # ====== YOUR CODE: ======
        device = next(self.parameters()).device
        z = z.to(device)

        # Convert latent z to features h with a linear layer.
        h_features = self.w_hhat(z)
        h_features = h_features.view(h_features.size(0), *self.features_shape)

        # Apply features decoder:
        x_reconstructed = self.features_decoder(h_features)
        # ========================

        # Scale to [-1, 1] (same dynamic range as original images).
        return torch.tanh(x_reconstructed)

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
            samples = self.decode(torch.randn(size=(n, self.z_dim), device=device))
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
    loss, data_loss, kldiv_loss = None, None, None

    # ====== YOUR CODE: ======

    # Flattens the inputs
    x = x.view(x.shape[0], -1)
    xr = xr.view(xr.shape[0], -1)

    # Gets the dimensions:
    dz = z_mu.shape[1]
    dx = x.shape[1]

    # Calculates the data-reconstruction loss:
    data_loss = torch.mean(torch.sum((x - xr) ** 2,  dim=1)/(x_sigma2 * dx))

    # Calculates the kldiv loss:
    z_sigma2 = torch.exp(z_log_sigma2)
    tr_sigma = torch.sum(z_sigma2, dim=1)
    log_det = torch.log(torch.prod(z_sigma2, dim=1))

    kldiv_loss = torch.mean(tr_sigma + torch.sum(z_mu ** 2, dim=1) - dz - log_det, dim=0)

    # Calculates the loss:
    loss = data_loss + kldiv_loss


    # ========================

    return loss, data_loss, kldiv_loss
