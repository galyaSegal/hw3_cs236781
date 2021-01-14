import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable
from torch.utils.data import DataLoader
from torch.optim.optimizer import Optimizer


class Discriminator(nn.Module):
    def __init__(self, in_size):
        """
        :param in_size: The size of on input image (without batch dimension).
        """
        super().__init__()
        self.in_size = in_size
        # TODO: Create the discriminator model layers.
        #  To extract image features you can use the EncoderCNN from the VAE
        #  section or implement something new.
        #  You can then use either an affine layer or another conv layer to
        #  flatten the features.
        # ====== YOUR CODE: ======
        # Based on 'Unsupervised Representation Learning with Deep Convolutional
        # Generative Adversarial Networks, Alec Redford and Luke Metz
        modules = []
        channels = [in_size[0], 128, 256, 512, 1024]
        num_layers = len(channels)
        kernel_sizes = [5] * (num_layers-1)
        padding = 2
        dilation = 1
        batch_norm = True
        dropout = 0
        strides = [2] * (num_layers-1)
        activation_param = 0.2

        # Create the layers
        for channel_idx in range(num_layers-1):
            in_features, out_features, kernel_size, stride = \
                channels[channel_idx], channels[channel_idx+1], kernel_sizes[channel_idx], strides[channel_idx]

            layer = nn.Conv2d(in_channels=in_features, out_channels=out_features,
                              kernel_size=kernel_size, padding=padding, dilation=dilation, stride=stride)
            modules.append(layer)

            if batch_norm and 0 < channel_idx:
                batch_norm_layer = nn.BatchNorm2d(num_features=out_features)
                modules.append(batch_norm_layer)

            activation_layer = nn.LeakyReLU(activation_param)
            modules.append(activation_layer)

            if channel_idx % 2 == 0 and dropout > 0:
                dropout_layer = nn.Dropout2d(p=dropout)
                modules.append(dropout_layer)

        self.features_extractor = nn.Sequential(*modules)

        # Classifier
        modules = []
        flatten_layer = nn.Flatten()
        modules.append(flatten_layer)
        in_features = channels[-1] * 4 * 4
        linear_layer = nn.Linear(in_features=in_features, out_features=1)
        modules.append(linear_layer)
        self.classifier = nn.Sequential(*modules)
        # ========================

    def forward(self, x):
        """
        :param x: Input of shape (N,C,H,W) matching the given in_size.
        :return: Discriminator class score (not probability) of
        shape (N,).
        """
        # TODO: Implement discriminator forward pass.
        #  No need to apply sigmoid to obtain probability - we'll combine it
        #  with the loss due to improved numerical stability.
        # ====== YOUR CODE: ======
        features = self.features_extractor(x)
        y = self.classifier(features)
        # ========================
        return y


class Generator(nn.Module):
    def __init__(self, z_dim, featuremap_size=4, out_channels=3):
        """
        :param z_dim: Dimension of latent space.
        :featuremap_size: Spatial size of first feature map to create
        (determines output size). For example set to 4 for a 4x4 feature map.
        :out_channels: Number of channels in the generated image.
        """
        super().__init__()
        self.z_dim = z_dim

        # TODO: Create the generator model layers.
        #  To combine image features you can use the DecoderCNN from the VAE
        #  section or implement something new.
        #  You can assume a fixed image size.
        # ====== YOUR CODE: ======
        # Based on 'Unsupervised Representation Learning with Deep Convolutional
        # Generative Adversarial Networks, Alec Redford and Luke Metz
        channels = [1024, 512, 256, 128, out_channels]
        num_layers = len(channels)
        kernel_sizes = [5] * num_layers
        padding = 2
        dilation = 1
        batch_norm = True
        dropout = 0
        strides = [2] * num_layers
        output_padding = 1

        # Linear layer
        out_features = channels[0] * featuremap_size ** 2
        linear_layer = nn.Linear(in_features=z_dim, out_features=out_features)
        self.project = nn.Sequential(linear_layer)
        self.features_size = [1024, featuremap_size, featuremap_size]

        # Generator
        modules = []
        for channel_idx in range(num_layers - 1):
            in_features, out_features, kernel_size, stride =\
                channels[channel_idx], channels[channel_idx + 1], kernel_sizes[channel_idx], strides[channel_idx]
            layer = nn.ConvTranspose2d(in_channels=in_features, out_channels=out_features,
                                       kernel_size=kernel_size, padding=padding, dilation=dilation,
                                       stride=stride, output_padding=output_padding)
            modules.append(layer)

            if batch_norm and channel_idx < num_layers-2:
                batch_norm_layer = nn.BatchNorm2d(num_features=out_features)
                modules.append(batch_norm_layer)

            if channel_idx < num_layers-2:
                activation_layer = nn.ReLU()
            else:
                # Tanh in last layer
                activation_layer = nn.Tanh()
            modules.append(activation_layer)

            if channel_idx % 2 == 0 and dropout > 0:
                dropout_layer = nn.Dropout2d(p=dropout)
                modules.append(dropout_layer)

        self.generator = nn.Sequential(*modules)
        # ========================

    def sample(self, n, with_grad=False):
        """
        Samples from the Generator.
        :param n: Number of instance-space samples to generate.
        :param with_grad: Whether the returned samples should be part of the
        generator's computation graph or standalone tensors (i.e. should be
        be able to backprop into them and compute their gradients).
        :return: A batch of samples, shape (N,C,H,W).
        """
        device = next(self.parameters()).device
        # TODO: Sample from the model.
        #  Generate n latent space samples and return their reconstructions.
        #  Don't use a loop.
        # ====== YOUR CODE: ======
        torch.autograd.set_grad_enabled(with_grad)
        samples = torch.randn(size=[n, self.z_dim],
                              device=device)
        samples = self.forward(samples)
        torch.autograd.set_grad_enabled(True)
        # ========================
        return samples

    def forward(self, z):
        """
        :param z: A batch of latent space samples of shape (N, latent_dim).
        :return: A batch of generated images of shape (N,C,H,W) which should be
        the shape which the Discriminator accepts.
        """
        # TODO: Implement the Generator forward pass.
        #  Don't forget to make sure the output instances have the same
        #  dynamic range as the original (real) images.
        # ====== YOUR CODE: ======
        batch_size = z.shape[0]
        projection = self.project(z)
        projection = torch.reshape(projection, [batch_size, *self.features_size])
        x = self.generator(projection)
        # ========================
        return x


def discriminator_loss_fn(y_data, y_generated, data_label=0, label_noise=0.0):
    """
    Computes the combined loss of the discriminator given real and generated
    data using a binary cross-entropy metric.
    This is the loss used to update the Discriminator parameters.
    :param y_data: Discriminator class-scores of instances of data sampled
    from the dataset, shape (N,).
    :param y_generated: Discriminator class-scores of instances of data
    generated by the generator, shape (N,).
    :param data_label: 0 or 1, label of instances coming from the real dataset.
    :param label_noise: The range of the noise to add. For example, if
    data_label=0 and label_noise=0.2 then the labels of the real data will be
    uniformly sampled from the range [-0.1,+0.1].
    :return: The combined loss of both.
    """
    assert data_label == 1 or data_label == 0
    # TODO:
    #  Implement the discriminator loss.
    #  See pytorch's BCEWithLogitsLoss for a numerically stable implementation.
    # ====== YOUR CODE: ======
    # Real data loss
    y_data_label = data_label * torch.ones_like(y_data)
    device = y_data_label.device
    if label_noise > 0:
        noise = torch.rand(y_data.shape) * label_noise - 0.5 * label_noise
        y_data_label += noise.to(device)
    loss_data = F.binary_cross_entropy_with_logits(input=y_data,
                                                   target=y_data_label)

    # Generated data loss
    y_generated_label = (1-data_label) * torch.ones_like(y_generated)
    if label_noise > 0:
        noise = torch.rand(y_data.shape) * label_noise - 0.5 * label_noise
        y_generated_label += noise.to(device)
    loss_generated = F.binary_cross_entropy_with_logits(input=y_generated,
                                                        target=y_generated_label)
    # ========================
    return loss_data + loss_generated


def generator_loss_fn(y_generated, data_label=0):
    """
    Computes the loss of the generator given generated data using a
    binary cross-entropy metric.
    This is the loss used to update the Generator parameters.
    :param y_generated: Discriminator class-scores of instances of data
    generated by the generator, shape (N,).
    :param data_label: 0 or 1, label of instances coming from the real dataset.
    :return: The generator loss.
    """
    assert data_label == 1 or data_label == 0
    # TODO:
    #  Implement the Generator loss.
    #  Think about what you need to compare the input to, in order to
    #  formulate the loss in terms of Binary Cross Entropy.
    # ====== YOUR CODE: ======
    y_generated_label = torch.ones_like(y_generated) * data_label
    loss = nn.functional.binary_cross_entropy_with_logits(input=y_generated,
                                                          target=y_generated_label)
    # ========================
    return loss


def train_batch(
    dsc_model: Discriminator,
    gen_model: Generator,
    dsc_loss_fn: Callable,
    gen_loss_fn: Callable,
    dsc_optimizer: Optimizer,
    gen_optimizer: Optimizer,
    x_data: DataLoader,
):
    """
    Trains a GAN for over one batch, updating both the discriminator and
    generator.
    :return: The discriminator and generator losses.
    """

    # TODO: Discriminator update
    #  1. Show the discriminator real and generated data
    #  2. Calculate discriminator loss
    #  3. Update discriminator parameters
    # ====== YOUR CODE: ======
    # Show the discriminator real and generated data
    real_data = x_data
    batch_size = real_data.shape[0]
    generated_data = gen_model.sample(n=batch_size, with_grad=True)

    # Calculate discriminator loss
    y_data = dsc_model(real_data)
    y_generated = dsc_model(generated_data)
    dsc_loss = dsc_loss_fn(y_data, y_generated)

    # Update discriminator parameters
    dsc_optimizer.zero_grad()
    dsc_loss.backward()
    dsc_optimizer.step()
    # ========================

    # TODO: Generator update
    #  1. Show the discriminator generated data
    #  2. Calculate generator loss
    #  3. Update generator parameters
    # ====== YOUR CODE: ======
    # Show the discriminator generated data
    generated_data = gen_model.sample(n=batch_size, with_grad=True)
    y_generated = dsc_model(generated_data)

    # Calculate generator loss
    gen_loss = gen_loss_fn(y_generated)

    # Update generator parameters
    gen_optimizer.zero_grad()
    gen_loss.backward()
    gen_optimizer.step()
    # ========================

    return dsc_loss.item(), gen_loss.item()


def save_checkpoint(gen_model, dsc_losses, gen_losses, checkpoint_file):
    """
    Saves a checkpoint of the generator, if necessary.
    :param gen_model: The Generator model to save.
    :param dsc_losses: Avg. discriminator loss per epoch.
    :param gen_losses: Avg. generator loss per epoch.
    :param checkpoint_file: Path without extension to save generator to.
    """

    saved = False
    checkpoint_file = f"{checkpoint_file}.pt"

    # TODO:
    #  Save a checkpoint of the generator model. You can use torch.save().
    #  You should decide what logic to use for deciding when to save.
    #  If you save, set saved to True.
    # ====== YOUR CODE: ======
    last_gen_loss = gen_losses[-1]
    last_dsc_loss = dsc_losses[-1]
    if len(gen_losses) == 1:
        saved = False
        return saved

    elif last_gen_loss < gen_losses[-2] and last_dsc_loss <= dsc_losses[-2]:
        torch.save(gen_model, checkpoint_file)
        saved = True
    # ========================

    return saved
