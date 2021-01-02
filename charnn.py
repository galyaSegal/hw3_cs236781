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
    # ====== YOUR CODE: ======
    # Sorts the text by a lexical order
    text = sorted(text)

    char_to_idx = dict()
    idx_to_char = dict()

    idx = 0
    for char in text:
        # A condition to avoid duplicated characters
        if char not in char_to_idx:
            char_to_idx[char] = idx
            idx_to_char[idx] = char
            idx += 1

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
    # ====== YOUR CODE: ======
    text_clean = text

    for char in chars_to_remove:
        if char in text_clean:
            text_clean = text_clean.replace(char, "")

    n_removed = len(text) - len(text_clean)
    # ========================
    return text_clean, n_removed


def chars_to_onehot(text: str, char_to_idx: dict) -> Tensor:
    """
    Embed a sequence of chars as a a tensor containing the one-hot encoding
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
    # ====== YOUR CODE: ======

    D = len(char_to_idx)
    N = len(text)

    # Initialize the one hot sequence tensor with zeros
    result = torch.zeros(N, D, dtype=torch.int8)

    # Ignite the one hot sequence tensor's rows
    for i, char in enumerate(text):
        result[i, char_to_idx[char]] = 1

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
    # ====== YOUR CODE: ======
    # Initialize the result string
    result = ""

    # Gets the indices that corresponds to the one hot representations
    reversed_one_hot = torch.argmax(embedded_text, dim=1)

    # Builds the result string
    for idx in reversed_one_hot:
        result += idx_to_char[idx.item()]
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
    :param char_to_idx: The mapping to create and embedding with.
    :param seq_len: The sequence length of each sample and label.
    :param device: The device on which to create the result tensors.
    :return: A tuple containing two tensors:
    samples, of shape (N, S, V) and labels of shape (N, S) where N is
    the number of created samples, S is the seq_len and V is the embedding
    dimension.
    """
    # ====== YOUR CODE: ======
    S = seq_len
    V = len(char_to_idx)
    N = (len(text) - 1) // S

    embedded_text = chars_to_onehot(text, char_to_idx)

    labels = torch.argmax(embedded_text, dim=1).to(device)
    # Shifts the labels to the left and reshape it to fit the samples
    labels = labels[1:S * N + 1].view(N, S)

    # Truncate and reshape the embedded text into sampled sequences
    samples = embedded_text[:S * N, :].view(N, S, V).to(device)

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
    softmax = nn.Softmax(dim=dim)
    y = y / temperature
    result = softmax(y)
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

    # ====== YOUR CODE: ======
    with torch.no_grad():
        # Calculates the number of characters to generate
        n_gen_chars = n_chars - len(start_sequence)

        input = chars_to_onehot(start_sequence, char_to_idx).unsqueeze(dim=0).to(device, dtype=torch.float)
        h = None
        for char in range(n_gen_chars):

            # Generates a new output given the previous hidden state
            outputs, h = model(input, hidden_state=h)
            chars_distribution = hot_softmax(outputs, dim=2, temperature=T)[:, -1, :]

            # Sample from the predicted distribution and convert it into a character
            predicted = idx_to_char[torch.multinomial(chars_distribution, 1)[0].item()]

            out_text += predicted
            input = chars_to_onehot(predicted, char_to_idx).unsqueeze(dim=0).to(device, dtype=torch.float)
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
        idx = []  # idx should be a 1-d list of indices.
        # ====== YOUR CODE: ======
        n_samples = len(self.dataset)
        # Calculates the number of batches
        n_batches = n_samples // self.batch_size

        # Builds a contiguous batches' indices list
        for i in range(n_batches):
            idx.extend([ix for ix in range(i, n_batches * self.batch_size, n_batches)])

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
        :param out_dim: Number of input dimensions (at each timestep).
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

        # ====== YOUR CODE: ======
        self.dropout = nn.Dropout(dropout)
        self.y = nn.Linear(self.h_dim, self.out_dim)

        # creates the gates components/modules for each layer
        for layer_idx in range(self.n_layers):
            if layer_idx != 0:
                gates = self.build_gates(self.h_dim)
            else:
                gates = self.build_gates(self.in_dim)

            # Registers the created sub modules' parameters
            for gate, modules in gates.items():
                for module_name, module in modules.items():
                    self.add_module(name="W_{}{}_{}".format(module_name, gate, layer_idx), module=module)

            self.layer_params.append(gates)
        # ========================

    def build_gates(self, in_dim):
        """Build a GRU's gates sub_modules"""

        activations = {"z": nn.Sigmoid(), "r": nn.Sigmoid(), "g": nn.Tanh()}
        gates = {
            gate: {
                "x": nn.Linear(in_features=in_dim,
                               out_features=self.h_dim, bias=False),
                "h": nn.Linear(in_features=self.h_dim,
                               out_features=self.h_dim),
                "activation": activations[gate]
            } for gate in ["z", "r", "g"]
        }

        return gates

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
        layer_output = []

        # ====== YOUR CODE: ======

        for t_step in range(seq_len):
            x_t = layer_input[:, t_step, :]

            for layer_idx, params in enumerate(self.layer_params):
                # Gets the previous h_t
                h_t = layer_states[layer_idx]

                z_params, r_params, g_params = params.values()

                # Activates the gates
                z_t = z_params["activation"](z_params["x"](x_t) + z_params["h"](h_t))
                r_t = r_params["activation"](r_params["x"](x_t) + r_params["h"](h_t))
                g_t = g_params["activation"](g_params["x"](x_t) + g_params["h"](r_t * h_t))

                h_t = self.dropout(z_t * h_t + (1 - z_t) * g_t)

                # Propagates h_t through time/layers
                x_t = h_t
                layer_states[layer_idx] = h_t

            # Saves the output for the current time step
            layer_output.append(self.y(x_t))

        layer_output = torch.stack(layer_output, dim=1)
        hidden_state = torch.stack(layer_states, dim=1)

        # ========================
        return layer_output, hidden_state
