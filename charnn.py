import re
import torch
import torch.nn as nn
import torch.utils.data
from torch import Tensor
from typing import Iterator


def char_maps(text: str):
    """
    Create mapping from the unique chars in a text to integers and
    vice-versa.
    :param text: Some text.
    :return: Two maps.
        - char_to_idx, a mapping from a character to a unique
        integer from zero to the number of unique chars in the text.
        - idx_to_char, a mapping from an index to the character
        represented by it. The reverse of the above map.

    """
    # TODO:
    #  Create two maps as described in the docstring above.
    #  It's best if you also sort the chars before assigning indices, so that
    #  they're in lexical order.
    # ====== YOUR CODE: ======
    char_list = list(set(text))
    char_list = sorted(sorted(char_list), key=str.upper)
    num_unique_chars = len(char_list)
    idx_list = list(range(num_unique_chars))

    idx_to_char = {x: char_list[x] for x in idx_list}
    char_to_idx = dict([(value, key) for key, value in idx_to_char.items()])
    # ========================
    return char_to_idx, idx_to_char


def remove_chars(text: str, chars_to_remove):
    """
    Removes all occurrences of the given chars from a text sequence.
    :param text: The text sequence.
    :param chars_to_remove: A list of characters that should be removed.
    :return:
        - text_clean: the text after removing the chars.
        - n_removed: Number of chars removed.
    """
    # TODO: Implement according to the docstring.
    # ====== YOUR CODE: ======
    text_clean = text
    for char in chars_to_remove:
        text_clean = text_clean.replace(char, '')
    n_removed = len(text) - len(text_clean)
    # ========================
    return text_clean, n_removed


def chars_to_onehot(text: str, char_to_idx: dict) -> Tensor:
    """
    Embed a sequence of chars as a tensor containing the one-hot encoding
    of each char. A one-hot encoding means that each char is represented as
    a tensor of zeros with a single '1' element at the index in the tensor
    corresponding to the index of that char.
    :param text: The text to embed.
    :param char_to_idx: Mapping from each char in the sequence to it's
    unique index.
    :return: Tensor of shape (N, D) where N is the length of the sequence
    and D is the number of unique chars in the sequence. The dtype of the
    returned tensor will be torch.int8.
    """
    # TODO: Implement the embedding.
    # ====== YOUR CODE: ======
    len_seq = len(text)
    num_chars = len(char_to_idx)
    result = torch.zeros(len_seq, num_chars, dtype=torch.int8)

    seq_idx = 0
    for char in text:
        char_idx = char_to_idx[char]
        result[seq_idx, char_idx] = 1
        seq_idx += 1

    # ========================
    return result


def onehot_to_chars(embedded_text: Tensor, idx_to_char: dict) -> str:
    """
    Reverses the embedding of a text sequence, producing back the original
    sequence as a string.
    :param embedded_text: Text sequence represented as a tensor of shape
    (N, D) where each row is the one-hot encoding of a character.
    :param idx_to_char: Mapping from indices to characters.
    :return: A string containing the text sequence represented by the
    embedding.
    """
    # TODO: Implement the reverse-embedding.
    # ====== YOUR CODE: ======
    result = ''
    len_seq = embedded_text.shape[0]
    for idx in range(len_seq):
        char_idx = (embedded_text[idx, :] == 1).nonzero().item()
        char = idx_to_char[char_idx]
        result += char
    # ========================
    return result


def chars_to_labelled_samples(text: str, char_to_idx: dict, seq_len: int, device="cpu"):
    """
    Splits a char sequence into smaller sequences of labelled samples.
    A sample here is a sequence of seq_len embedded chars.
    Each sample has a corresponding label, which is also a sequence of
    seq_len chars represented as indices. The label is constructed such that
    the label of each char is the next char in the original sequence.
    :param text: The char sequence to split.
    :param char_to_idx: The mapping to create an embedding with.
    :param seq_len: The sequence length of each sample and label.
    :param device: The device on which to create the result tensors.
    :return: A tuple containing two tensors:
    samples, of shape (N, S, V) and labels of shape (N, S) where N is
    the number of created samples, S is the seq_len and V is the embedding
    dimension.
    """
    # TODO:
    #  Implement the labelled samples creation.
    #  1. Embed the given text.
    #  2. Create the samples tensor by splitting to groups of seq_len.
    #     Notice that the last char has no label, so don't use it.
    #  3. Create the labels tensor in a similar way and convert to indices.
    #  Note that no explicit loops are required to implement this function.
    # ====== YOUR CODE: ======
    embed_text = chars_to_onehot(text=text, char_to_idx=char_to_idx)  # one hot tensor of size (N, V)
    num_inx_to_use = embed_text.shape[0] - ((embed_text.shape[0]-1) % seq_len)
    num_samples = int(num_inx_to_use / seq_len)

    cut_embed_text = embed_text[:num_inx_to_use-1, :]

    samples = torch.reshape(cut_embed_text, shape=(num_samples, seq_len, -1)).to(device)
    labels = torch.nonzero(embed_text[1:num_inx_to_use, :])[:, 1]
    labels = labels.reshape(num_samples, seq_len).to(device)
    # ========================
    return samples, labels


def hot_softmax(y, dim=0, temperature=1.0):
    """
    A softmax which first scales the input by 1/temperature and
    then computes softmax along the given dimension.
    :param y: Input tensor.
    :param dim: Dimension to apply softmax on.
    :param temperature: Temperature.
    :return: Softmax computed with the temperature parameter.
    """
    # TODO: Implement based on the above.
    # ====== YOUR CODE: ======
    assert temperature != 0, 'Division by zero'

    scaled_y = y / temperature
    softmax = torch.nn.Softmax(dim=dim)
    result = softmax(scaled_y)
    # ========================
    return result


def generate_from_model(model, start_sequence, n_chars, char_maps, T):
    """
    Generates a sequence of chars based on a given model and a start sequence.
    :param model: An RNN model. forward should accept (x,h0) and return (y,
    h_s) where x is an embedded input sequence, h0 is an initial hidden state,
    y is an embedded output sequence and h_s is the final hidden state.
    :param start_sequence: The initial sequence to feed the model.
    :param n_chars: The total number of chars to generate (including the
    initial sequence).
    :param char_maps: A tuple as returned by char_maps(text).
    :param T: Temperature for sampling with softmax-based distribution.
    :return: A string starting with the start_sequence and continuing for
    with chars predicted by the model, with a total length of n_chars.
    """
    assert len(start_sequence) < n_chars
    device = next(model.parameters()).device
    char_to_idx, idx_to_char = char_maps
    out_text = start_sequence

    # TODO:
    #  Implement char-by-char text generation.
    #  1. Feed the start_sequence into the model.
    #  2. Sample a new char from the output distribution of the last output
    #     char. Convert output to probabilities first.
    #     See torch.multinomial() for the sampling part.
    #  3. Feed the new char into the model.
    #  4. Rinse and Repeat.
    #  Note that tracking tensor operations for gradient calculation is not
    #  necessary for this. Best to disable tracking for speed.
    #  See torch.no_grad().
    # ====== YOUR CODE: ======
    sequence = start_sequence
    n_chars_to_gen = n_chars - len(start_sequence)

    with torch.no_grad():  # disable unnecessary gradient tracking for speed
        h_s = None
        # Loop over number of chars needed to be generated
        for char_idx in range(n_chars_to_gen):
            input = torch.unsqueeze(input=chars_to_onehot(text=sequence, char_to_idx=char_to_idx), dim=0).to(
                dtype=torch.float, device=device)
            y, h_s = model(input, h_s)

            # Generate probabilities
            out_prob = hot_softmax(y=y[0, -1, :], temperature=T)  # distribution of the last output char

            # Sample from the generated distribution
            sample_idx = torch.multinomial(input=out_prob, num_samples=1)[0]
            sample_char = idx_to_char[sample_idx.item()]

            # Update the sequence and the out_text
            sequence = sample_char
            out_text = out_text + sample_char
    # ========================

    return out_text


class SequenceBatchSampler(torch.utils.data.Sampler):
    """
    Samples indices from a dataset containing consecutive sequences.
    This sample ensures that samples in the same index of adjacent
    batches are also adjacent in the dataset.
    """

    def __init__(self, dataset: torch.utils.data.Dataset, batch_size):
        """
        :param dataset: The dataset for which to create indices.
        :param batch_size: Number of indices in each batch.
        """
        super().__init__(dataset)
        self.dataset = dataset
        self.batch_size = batch_size

    def __iter__(self) -> Iterator[int]:
        # TODO:
        #  Return an iterator of indices, i.e. numbers in range(len(dataset)).
        #  dataset and represents one  batch.
        #  The indices must be generated in a way that ensures
        #  that when a batch of size self.batch_size of indices is taken, samples in
        #  the same index of adjacent batches are also adjacent in the dataset.
        #  In the case when the last batch can't have batch_size samples,
        #  you can drop it.
        idx = None  # idx should be a 1-d list of indices.
        # ====== YOUR CODE: ======
        num_batches = len(self.dataset) // self.batch_size
        tot_num_samples = num_batches * self.batch_size
        idx_list = torch.tensor(list(range(tot_num_samples)))
        idx = torch.transpose(torch.reshape(idx_list, (self.batch_size, num_batches)), 0, 1)
        idx = torch.reshape(idx, (1, -1)).squeeze(dim=0).tolist()
        # ========================
        return iter(idx)

    def __len__(self):
        return len(self.dataset)


class MultilayerGRU(nn.Module):
    """
    Represents a multi-layer GRU (gated recurrent unit) model.
    """

    def __init__(self, in_dim, h_dim, out_dim, n_layers, dropout=0):
        """
        :param in_dim: Number of input dimensions (at each timestep).
        :param h_dim: Number of hidden state dimensions.
        :param out_dim: Number of output dimensions (at each timestep).
        :param n_layers: Number of layer in the model.
        :param dropout: Level of dropout to apply between layers. Zero
        disables.
        """
        super().__init__()
        assert in_dim > 0 and h_dim > 0 and out_dim > 0 and n_layers > 0

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.h_dim = h_dim
        self.n_layers = n_layers
        self.layer_params = []

        # TODO: Create the parameters of the model for all layers.
        #  To implement the affine transforms you can use either nn.Linear
        #  modules (recommended) or create W and b tensor pairs directly.
        #  Create these modules or tensors and save them per-layer in
        #  the layer_params list.
        #  Important note: You must register the created parameters so
        #  they are returned from our module's parameters() function.
        #  Usually this happens automatically when we assign a
        #  module/tensor as an attribute in our module, but now we need
        #  to do it manually since we're not assigning attributes. So:
        #    - If you use nn.Linear modules, call self.add_module() on them
        #      to register each of their parameters as part of your model.
        #    - If you use tensors directly, wrap them in nn.Parameter() and
        #      then call self.register_parameter() on them. Also make
        #      sure to initialize them. See functions in torch.nn.init.
        # ====== YOUR CODE: ======
        self.dropout = dropout
        self.out_dim = out_dim

        for layer_idx in range(n_layers):
            in_dim = in_dim if layer_idx == 0 else h_dim

            # Define layer params
            fc_xz = nn.Linear(in_features=in_dim, out_features=h_dim, bias=False)
            fc_hz = nn.Linear(in_features=h_dim, out_features=h_dim, bias=True)
            fc_xr = nn.Linear(in_features=in_dim, out_features=h_dim, bias=False)
            fc_hr = nn.Linear(in_features=h_dim, out_features=h_dim, bias=True)
            fc_xg = nn.Linear(in_features=in_dim, out_features=h_dim, bias=False)
            fc_hg = nn.Linear(in_features=h_dim, out_features=h_dim, bias=True)
            sigmoid = nn.Sigmoid()
            tanh = nn.Tanh()

            # Define a dictionary of layer params
            layer_params = {'fc_xz_{}'.format(layer_idx): fc_xz,
                            'fc_hz_{}'.format(layer_idx): fc_hz,
                            'fc_xr_{}'.format(layer_idx): fc_xr,
                            'fc_hr_{}'.format(layer_idx): fc_hr,
                            'fc_xg_{}'.format(layer_idx): fc_xg,
                            'fc_hg_{}'.format(layer_idx): fc_hg,
                            'sigmoid_{}'.format(layer_idx): sigmoid,
                            'tanh_{}'.format(layer_idx): tanh}
            if dropout > 0:
                layer_params['dropout_{}'.format(layer_idx)] = nn.Dropout2d(p=dropout)

            for key, value in layer_params.items():
                self.add_module(name=key, module=value)
            self.layer_params.append(layer_params)

        # Add output layer
        fc_hy = nn.Linear(in_features=h_dim, out_features=out_dim, bias=True)
        last_layer_params = {'fc_hy': fc_hy}
        self.layer_params.append(last_layer_params)
        self.add_module(name='fc_hy', module=fc_hy)
        # ========================

    def forward(self, input: Tensor, hidden_state: Tensor = None):
        """
        :param input: Batch of sequences. Shape should be (B, S, I) where B is
        the batch size, S is the length of each sequence and I is the
        input dimension (number of chars in the case of a char RNN).
        :param hidden_state: Initial hidden state per layer (for the first
        char). Shape should be (B, L, H) where B is the batch size, L is the
        number of layers, and H is the number of hidden dimensions.
        :return: A tuple of (layer_output, hidden_state).
        The layer_output tensor is the output of the last RNN layer,
        of shape (B, S, O) where B,S are as above and O is the output
        dimension.
        The hidden_state tensor is the final hidden state, per layer, of shape
        (B, L, H) as above.
        """
        batch_size, seq_len, _ = input.shape

        layer_states = []
        for i in range(self.n_layers):
            if hidden_state is None:
                layer_states.append(
                    torch.zeros(batch_size, self.h_dim, device=input.device)
                )
            else:
                layer_states.append(hidden_state[:, i, :])

        layer_input = input

        # TODO:
        #  Implement the model's forward pass.
        #  You'll need to go layer-by-layer from bottom to top (see diagram).
        #  Tip: You can use torch.stack() to combine multiple tensors into a
        #  single tensor in a differentiable manner.
        # ====== YOUR CODE: ======
        assert self.n_layers > 0, 'number of layers is zero'

        # Initialize layer_output and hidden_state
        layer_output = torch.zeros(batch_size, seq_len, self.out_dim)

        # Loop over time (through the sequence)
        for char_idx in range(seq_len):
            xt = layer_input[:, char_idx, :]

            # Loop over layers
            for layer_idx in range(self.n_layers):
                hidden_layer = layer_states[layer_idx]
                # Extract layer params
                fc_xz = self.layer_params[layer_idx]['fc_xz_{}'.format(layer_idx)]
                fc_hz = self.layer_params[layer_idx]['fc_hz_{}'.format(layer_idx)]
                fc_xr = self.layer_params[layer_idx]['fc_xr_{}'.format(layer_idx)]
                fc_hr = self.layer_params[layer_idx]['fc_hr_{}'.format(layer_idx)]
                fc_xg = self.layer_params[layer_idx]['fc_xg_{}'.format(layer_idx)]
                fc_hg = self.layer_params[layer_idx]['fc_hg_{}'.format(layer_idx)]
                sigmoid = self.layer_params[layer_idx]['sigmoid_{}'.format(layer_idx)]
                tanh = self.layer_params[layer_idx]['tanh_{}'.format(layer_idx)]

                zt = sigmoid(fc_xz(xt) + fc_hz(hidden_layer))
                rt = sigmoid(fc_xr(xt) + fc_hr(hidden_layer))
                gt = tanh(fc_xg(xt) + fc_hg(rt * hidden_layer))
                hidden_layer = zt * hidden_layer + (1-zt) * gt

                if layer_idx > 0 and self.dropout > 0:
                    dropout_layer = self.modules['dropout_{}'.format(layer_idx)]
                    hidden_layer = dropout_layer(hidden_layer)

                layer_states[layer_idx] = hidden_layer
                xt = hidden_layer

            fc_hy = self.layer_params[self.n_layers]['fc_hy']
            layer_output[:, char_idx, :] = fc_hy(hidden_layer)

        hidden_state = torch.stack(layer_states, dim=1)
        # ========================
        return layer_output, hidden_state
