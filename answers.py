r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""

# ==============
# Part 1 answers


def part1_rnn_hyperparams():
    hypers = dict(
        batch_size=10,
        seq_len=20,
        h_dim=32,
        n_layers=3,
        dropout=0.2,
        learn_rate=0.01,
        lr_sched_factor=0.0,
        lr_sched_patience=20,
    )
    # TODO: Set the hyperparameters to train the model.
    # ====== YOUR CODE: ======
    # ========================
    return hypers


def part1_generation_params():
    start_seq = ""
    temperature = 0.0001
    # TODO: Tweak the parameters to generate a literary masterpiece.
    # ====== YOUR CODE: ======
    raise NotImplementedError()
    # ========================
    return start_seq, temperature


part1_q1 = r"""
**Your answer:**
We use sequences for two main reasons:
1. It allows us to train in batches, which is more efficient and therefore is much faster.
2. Storing the computational graphs is required only through a single sequence. Calculating the gradients through the 
whole text is impractical and also may cause exploding / vanishing gradients. 
"""

part1_q2 = r"""
**Your answer:**
During the generation process, the hidden state is propagated through the whole process and so longer 'memory' is 
established.

"""

part1_q3 = r"""
**Your answer:**
The batches were selected in a way that causality is maintained, and therefore we manage to improve logical causation
 during the training process. Shuffling would ruin that order.
"""

part1_q4 = r"""
**Your answer:**
1. During the training process, we aim to learn the generalized distribution of the chars in the text and therefore 
choose the temperature to be 1.0. \n
When sampling during the generation process, we wish to choose the char which is the most probable. Lowering the 
temperature results in greater differences between the probabilities, which allows a more significant selection of the
 next character.
2. A high temperature will lower the differences in the probabilities of each char. Very high temperature can result in 
practically uniform distribution, and thus the learning process will have very little effect on the chars generated. \n
The text generated with T > 0.8 was less coherent, but looked more like a play.
3. A low temperature will result in greater differences in the probabilities. The text generated with T < 0.2 was more 
coherent word-wise, but looked less like a play. Moreover, the word 'the' appeared significantly more times. We assume
that when the temperature is very low, the model tends to choose the most probable and frequent words, like 'the'. On 
the other hand, less frequent words which define the structure of the play and are much less probable don't appear. 
"""
# ==============


# ==============
# Part 2 answers

PART2_CUSTOM_DATA_URL = None


def part2_vae_hyperparams():
    hypers = dict(
        batch_size=16,
        h_dim=512,
        z_dim=32,
        x_sigma2=0.001,
        learn_rate=0.0001,
        betas=(0.5, 0.5),
    )
    # TODO: Tweak the hyperparameters to generate a former president.
    # ====== YOUR CODE: ======

    # ========================
    return hypers


part2_q1 = r"""
**Your answer:**
The $\sigma^2$ hyperparameter normalizes the data loss term. \n
Small values cause large 
"""

part2_q2 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part2_q3 = r"""
**Your answer:**
We wish to create a reconstructed image such that the probability of sampling it from the original image set
 will be maximal. \n
In other words, we wish to create images with the maximum likelihood of being sampled from the original image space. 
"""

part2_q4 = r"""
**Your answer:**
Since the variance is usually small and is within the interval od [0,1], the log transform allows mapping it into 
a much greater range of [-inf, 0]. As a result, the optimization algorithm can achieve better results. 
In other words, the training is more efficient when modeling the log of the variance. 
"""

# ==============

# ==============
# Part 3 answers

PART3_CUSTOM_DATA_URL = None


def part3_gan_hyperparams():
    hypers = dict(
        batch_size=16,
        z_dim=128,
        data_label=0,
        label_noise=0.02,
        discriminator_optimizer=dict(
            type="SGD",  # Any name in nn.optim like SGD, Adam
            lr=0.001,
            momentum=0.5
            # You an add extra args for the optimizer here
        ),
        generator_optimizer=dict(
            type="Adam",  # Any name in nn.optim like SGD, Adam
            lr=0.001,
            betas=(0.5, 0.5)
            # You an add extra args for the optimizer here
        ),
    )
    # TODO: Tweak the hyperparameters to train your GAN.
    # ====== YOUR CODE: ======

    # ========================
    return hypers


part3_q1 = r"""
**Your answer:**
In each batch during the training, the generator generates new samples in order to calculate
the loss and update the parameters of both the generator model and the descriminator model.
In order for gradient calculation, we sample from the GAN while maintaining gradients.
When visualising the generated samples during the training, no gradients need to be 
accumulated and so we sample from the GAN without maintaining gradients.

"""

part3_q2 = r"""
**Your answer:**
1. No, because if the discriminator is poorly trained, the generator will be able to 'fool'
it while actually generating bad images. The generator loss on its own is not enough for 
evaluating the quality of the generated images.
2. It means that the discriminator converged to some minimum (i.e. the discriminator is 
doing the best it can), while the generator is improving.

"""

part3_q3 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

# ==============
