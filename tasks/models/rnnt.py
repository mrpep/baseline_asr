import math
from typing import Callable, Dict, List, Optional, Tuple

import torch
from torchaudio.models import Conformer, RNNT
from torchaudio.models.rnnt import _Joiner, _Predictor, _TimeReduction, _Transcriber

from tqdm import tqdm

TrieNode = Tuple[Dict[int, "TrieNode"], int, Optional[Tuple[int, int]]]

class _ConformerEncoder(torch.nn.Module, _Transcriber):
    def __init__(
        self,
        *,
        input_dim: int,
        output_dim: int,
        time_reduction_stride: int,
        conformer_input_dim: int,
        conformer_ffn_dim: int,
        conformer_num_layers: int,
        conformer_num_heads: int,
        conformer_depthwise_conv_kernel_size: int,
        conformer_dropout: float,
    ) -> None:
        super().__init__()
        self.time_reduction = _TimeReduction(time_reduction_stride)
        self.input_linear = torch.nn.Linear(input_dim * time_reduction_stride, conformer_input_dim)
        self.conformer = Conformer(
            num_layers=conformer_num_layers,
            input_dim=conformer_input_dim,
            ffn_dim=conformer_ffn_dim,
            num_heads=conformer_num_heads,
            depthwise_conv_kernel_size=conformer_depthwise_conv_kernel_size,
            dropout=conformer_dropout,
            use_group_norm=True,
            convolution_first=True,
        )
        self.output_linear = torch.nn.Linear(conformer_input_dim, output_dim)
        self.layer_norm = torch.nn.LayerNorm(output_dim)

    def forward(self, input: torch.Tensor, lengths: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        time_reduction_out, time_reduction_lengths = self.time_reduction(input, lengths)
        input_linear_out = self.input_linear(time_reduction_out)
        x, lengths = self.conformer(input_linear_out, time_reduction_lengths)
        output_linear_out = self.output_linear(x)
        layer_norm_out = self.layer_norm(output_linear_out)
        return layer_norm_out, lengths

    def infer(
        self,
        input: torch.Tensor,
        lengths: torch.Tensor,
        states: Optional[List[List[torch.Tensor]]],
    ) -> Tuple[torch.Tensor, torch.Tensor, List[List[torch.Tensor]]]:
        raise RuntimeError("Conformer does not support streaming inference.")


class _JoinerBiasing(torch.nn.Module):
    r"""Recurrent neural network transducer (RNN-T) joint network.

    Args:
        input_dim (int): source and target input dimension.
        output_dim (int): output dimension.
        activation (str, optional): activation function to use in the joiner.
            Must be one of ("relu", "tanh"). (Default: "relu")
        biasing (bool): perform biasing
        deepbiasing (bool): perform deep biasing
        attndim (int): dimension of the biasing vector hptr

    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        activation: str = "relu",
        biasing: bool = False,
        deepbiasing: bool = False,
        attndim: int = 1,
    ) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim, bias=True)
        self.biasing = biasing
        self.deepbiasing = deepbiasing
        if self.biasing and self.deepbiasing:
            self.biasinglinear = torch.nn.Linear(attndim, input_dim, bias=True)
            self.attndim = attndim
        if activation == "relu":
            self.activation = torch.nn.ReLU()
        elif activation == "tanh":
            self.activation = torch.nn.Tanh()
        else:
            raise ValueError(f"Unsupported activation {activation}")

    def forward(
        self,
        source_encodings: torch.Tensor,
        source_lengths: torch.Tensor,
        target_encodings: torch.Tensor,
        target_lengths: torch.Tensor,
        hptr: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        r"""Forward pass for training.

        B: batch size;
        T: maximum source sequence length in batch;
        U: maximum target sequence length in batch;
        D: dimension of each source and target sequence encoding.

        Args:
            source_encodings (torch.Tensor): source encoding sequences, with
                shape `(B, T, D)`.
            source_lengths (torch.Tensor): with shape `(B,)` and i-th element representing
                valid sequence length of i-th batch element in ``source_encodings``.
            target_encodings (torch.Tensor): target encoding sequences, with shape `(B, U, D)`.
            target_lengths (torch.Tensor): with shape `(B,)` and i-th element representing
                valid sequence length of i-th batch element in ``target_encodings``.
            hptr (torch.Tensor): deep biasing vector with shape `(B, T, U, A)`.

        Returns:
            (torch.Tensor, torch.Tensor, torch.Tensor):
                torch.Tensor
                    joint network output, with shape `(B, T, U, output_dim)`.
                torch.Tensor
                    output source lengths, with shape `(B,)` and i-th element representing
                    number of valid elements along dim 1 for i-th batch element in joint network output.
                torch.Tensor
                    output target lengths, with shape `(B,)` and i-th element representing
                    number of valid elements along dim 2 for i-th batch element in joint network output.
                torch.Tensor
                    joint network second last layer output (i.e. before self.linear), with shape `(B, T, U, D)`.
        """
        joint_encodings = source_encodings.unsqueeze(2).contiguous() + target_encodings.unsqueeze(1).contiguous()
        if self.biasing and self.deepbiasing and hptr is not None:
            hptr = self.biasinglinear(hptr)
            joint_encodings += hptr
        elif self.biasing and self.deepbiasing:
            # Hack here for unused parameters
            joint_encodings += self.biasinglinear(joint_encodings.new_zeros(1, self.attndim)).mean() * 0
        activation_out = self.activation(joint_encodings)
        output = self.linear(activation_out)
        return output, source_lengths, target_lengths, activation_out


class RNNTBiasing(RNNT):
    r"""torchaudio.models.RNNT()

    Recurrent neural network transducer (RNN-T) model.

    Note:
        To build the model, please use one of the factory functions.

    Args:
        transcriber (torch.nn.Module): transcription network.
        predictor (torch.nn.Module): prediction network.
        joiner (torch.nn.Module): joint network.
        attndim (int): TCPGen attention dimension
        biasing (bool): If true, use biasing, otherwise use standard RNN-T
        deepbiasing (bool): If true, use deep biasing by extracting the biasing vector
        embdim (int): dimension of symbol embeddings
        jointdim (int): dimension of the joint network joint dimension
        charlist (list): The list of word piece tokens in the same order as the output layer
        encoutdim (int): dimension of the encoder output vectors
        dropout_tcpgen (float): dropout rate for TCPGen
        tcpsche (int): The epoch at which TCPGen starts to train
        DBaverage (bool): If true, instead of TCPGen, use DBRNNT for biasing
    """

    def __init__(
        self,
        transcriber: _Transcriber,
        predictor: _Predictor,
        joiner: _Joiner,
        attndim: int,
        biasing: bool,
        deepbiasing: bool,
        embdim: int,
        jointdim: int,
        charlist: List[str],
        encoutdim: int,
        dropout_tcpgen: float,
        tcpsche: int,
        DBaverage: bool,
    ) -> None:
        super().__init__(transcriber, predictor, joiner)
        self.attndim = attndim
        self.deepbiasing = deepbiasing
        self.jointdim = jointdim
        self.embdim = embdim
        self.encoutdim = encoutdim
        self.char_list = charlist or []
        self.blank_idx = self.char_list.index("<blank>")
        self.nchars = len(self.char_list)
        self.DBaverage = DBaverage
        self.biasing = biasing
        if self.biasing:
            if self.deepbiasing and self.DBaverage:
                # Deep biasing without TCPGen
                self.biasingemb = torch.nn.Linear(self.nchars, self.attndim, bias=False)
            else:
                # TCPGen parameters
                self.ooKBemb = torch.nn.Embedding(1, self.embdim)
                self.Qproj_char = torch.nn.Linear(self.embdim, self.attndim)
                self.Qproj_acoustic = torch.nn.Linear(self.encoutdim, self.attndim)
                self.Kproj = torch.nn.Linear(self.embdim, self.attndim)
                self.pointer_gate = torch.nn.Linear(self.attndim + self.jointdim, 1)
        self.dropout_tcpgen = torch.nn.Dropout(dropout_tcpgen)
        self.tcpsche = tcpsche

    def forward(
        self,
        sources: torch.Tensor,
        source_lengths: torch.Tensor,
        targets: torch.Tensor,
        target_lengths: torch.Tensor,
        tries: TrieNode,
        current_epoch: int,
        predictor_state: Optional[List[List[torch.Tensor]]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[List[torch.Tensor]], torch.Tensor, torch.Tensor]:
        r"""Forward pass for training.

        B: batch size;
        T: maximum source sequence length in batch;
        U: maximum target sequence length in batch;
        D: feature dimension of each source sequence element.

        Args:
            sources (torch.Tensor): source frame sequences right-padded with right context, with
                shape `(B, T, D)`.
            source_lengths (torch.Tensor): with shape `(B,)` and i-th element representing
                number of valid frames for i-th batch element in ``sources``.
            targets (torch.Tensor): target sequences, with shape `(B, U)` and each element
                mapping to a target symbol.
            target_lengths (torch.Tensor): with shape `(B,)` and i-th element representing
                number of valid frames for i-th batch element in ``targets``.
            tries (TrieNode): wordpiece prefix trees representing the biasing list to be searched
            current_epoch (Int): the current epoch number to determine if TCPGen should be trained
                at this epoch
            predictor_state (List[List[torch.Tensor]] or None, optional): list of lists of tensors
                representing prediction network internal state generated in preceding invocation
                of ``forward``. (Default: ``None``)

        Returns:
            (torch.Tensor, torch.Tensor, torch.Tensor, List[List[torch.Tensor]]):
                torch.Tensor
                    joint network output, with shape
                    `(B, max output source length, max output target length, output_dim (number of target symbols))`.
                torch.Tensor
                    output source lengths, with shape `(B,)` and i-th element representing
                    number of valid elements along dim 1 for i-th batch element in joint network output.
                torch.Tensor
                    output target lengths, with shape `(B,)` and i-th element representing
                    number of valid elements along dim 2 for i-th batch element in joint network output.
                List[List[torch.Tensor]]
                    output states; list of lists of tensors
                    representing prediction network internal state generated in current invocation
                    of ``forward``.
                torch.Tensor
                    TCPGen distribution, with shape
                    `(B, max output source length, max output target length, output_dim (number of target symbols))`.
                torch.Tensor
                    Generation probability (or copy probability), with shape
                    `(B, max output source length, max output target length, 1)`.
        """
        source_encodings, source_lengths = self.transcriber(
            input=sources,
            lengths=source_lengths,
        )
        target_encodings, target_lengths, predictor_state = self.predictor(
            input=targets,
            lengths=target_lengths,
            state=predictor_state,
        )
        # Forward TCPGen
        hptr = None
        tcpgen_dist, p_gen = None, None
        if self.biasing and current_epoch >= self.tcpsche and tries != []:
            ptrdist_mask, p_gen_mask = self.get_tcpgen_step_masks(targets, tries)
            hptr, tcpgen_dist = self.forward_tcpgen(targets, ptrdist_mask, source_encodings)
            hptr = self.dropout_tcpgen(hptr)
        elif self.biasing:
            # Hack here to bypass unused parameters
            if self.DBaverage and self.deepbiasing:
                dummy = self.biasingemb(source_encodings.new_zeros(1, len(self.char_list))).mean()
            else:
                dummy = source_encodings.new_zeros(1, self.embdim)
                dummy = self.Qproj_char(dummy).mean()
                dummy += self.Qproj_acoustic(source_encodings.new_zeros(1, source_encodings.size(-1))).mean()
                dummy += self.Kproj(source_encodings.new_zeros(1, self.embdim)).mean()
                dummy += self.pointer_gate(source_encodings.new_zeros(1, self.attndim + self.jointdim)).mean()
                dummy += self.ooKBemb.weight.mean()
            dummy = dummy * 0
            source_encodings += dummy

        output, source_lengths, target_lengths, jointer_activation = self.joiner(
            source_encodings=source_encodings,
            source_lengths=source_lengths,
            target_encodings=target_encodings,
            target_lengths=target_lengths,
            hptr=hptr,
        )

        # Calculate Generation Probability
        if self.biasing and hptr is not None and tcpgen_dist is not None:
            p_gen = torch.sigmoid(self.pointer_gate(torch.cat((jointer_activation, hptr), dim=-1)))
            # avoid collapsing to ooKB token in the first few updates
            # if current_epoch == self.tcpsche:
            #     p_gen = p_gen * 0.1
            p_gen = p_gen.masked_fill(p_gen_mask.bool().unsqueeze(1).unsqueeze(-1), 0)

        return (output, source_lengths, target_lengths, predictor_state, tcpgen_dist, p_gen)

    def get_tcpgen_distribution(self, query, ptrdist_mask):
        # Make use of the predictor embedding matrix
        keyvalues = torch.cat([self.predictor.embedding.weight.data, self.ooKBemb.weight], dim=0)
        keyvalues = self.dropout_tcpgen(self.Kproj(keyvalues))
        # B * T * U * attndim, nbpe * attndim -> B * T * U * nbpe
        tcpgendist = torch.einsum("ntuj,ij->ntui", query, keyvalues)
        tcpgendist = tcpgendist / math.sqrt(query.size(-1))
        ptrdist_mask = ptrdist_mask.unsqueeze(1).repeat(1, tcpgendist.size(1), 1, 1)
        tcpgendist.masked_fill_(ptrdist_mask.bool(), -1e9)
        tcpgendist = torch.nn.functional.softmax(tcpgendist, dim=-1)
        # B * T * U * nbpe, nbpe * attndim -> B * T * U * attndim
        hptr = torch.einsum("ntui,ij->ntuj", tcpgendist[:, :, :, :-1], keyvalues[:-1, :])
        return hptr, tcpgendist

    def forward_tcpgen(self, targets, ptrdist_mask, source_encodings):
        tcpgen_dist = None
        if self.DBaverage and self.deepbiasing:
            hptr = self.biasingemb(1 - ptrdist_mask[:, :, :-1].float()).unsqueeze(1)
        else:
            query_char = self.predictor.embedding(targets)
            query_char = self.Qproj_char(query_char).unsqueeze(1)  # B * 1 * U * attndim
            query_acoustic = self.Qproj_acoustic(source_encodings).unsqueeze(2)  # B * T * 1 * attndim
            query = query_char + query_acoustic  # B * T * U * attndim
            hptr, tcpgen_dist = self.get_tcpgen_distribution(query, ptrdist_mask)
        return hptr, tcpgen_dist

    def get_tcpgen_step_masks(self, yseqs, resettrie):
        seqlen = len(yseqs[0])
        batch_masks = yseqs.new_ones(len(yseqs), seqlen, len(self.char_list) + 1)
        p_gen_masks = []
        for i, yseq in enumerate(yseqs):
            new_tree = resettrie
            p_gen_mask = []
            for j, vy in enumerate(yseq):
                vy = vy.item()
                new_tree = new_tree[0]
                if vy in [self.blank_idx]:
                    new_tree = resettrie
                    p_gen_mask.append(0)
                elif self.char_list[vy].endswith("▁"):
                    if vy in new_tree and new_tree[vy][0] != {}:
                        new_tree = new_tree[vy]
                    else:
                        new_tree = resettrie
                    p_gen_mask.append(0)
                elif vy not in new_tree:
                    new_tree = [{}]
                    p_gen_mask.append(1)
                else:
                    new_tree = new_tree[vy]
                    p_gen_mask.append(0)
                batch_masks[i, j, list(new_tree[0].keys())] = 0
                # In the original paper, ooKB node was not masked
                # In this implementation, if not masking ooKB, ooKB probability
                # would quickly collapse to 1.0 in the first few updates.
                # Haven't found out why this happened.
                # batch_masks[i, j, -1] = 0
            p_gen_masks.append(p_gen_mask + [1] * (seqlen - len(p_gen_mask)))
        p_gen_masks = torch.Tensor(p_gen_masks).to(yseqs.device).byte()
        return batch_masks, p_gen_masks

    def get_tcpgen_step_masks_prefix(self, yseqs, resettrie):
        # Implemented for prefix-based wordpieces, not tested yet
        seqlen = len(yseqs[0])
        batch_masks = yseqs.new_ones(len(yseqs), seqlen, len(self.char_list) + 1)
        p_gen_masks = []
        for i, yseq in enumerate(yseqs):
            p_gen_mask = []
            new_tree = resettrie
            for j, vy in enumerate(yseq):
                vy = vy.item()
                new_tree = new_tree[0]
                if vy in [self.blank_idx]:
                    new_tree = resettrie
                    batch_masks[i, j, list(new_tree[0].keys())] = 0
                elif self.char_list[vy].startswith("▁"):
                    new_tree = resettrie
                    if vy not in new_tree[0]:
                        batch_masks[i, j, list(new_tree[0].keys())] = 0
                    else:
                        new_tree = new_tree[0][vy]
                        batch_masks[i, j, list(new_tree[0].keys())] = 0
                        if new_tree[1] != -1:
                            batch_masks[i, j, list(resettrie[0].keys())] = 0
                else:
                    if vy not in new_tree:
                        new_tree = resettrie
                        batch_masks[i, j, list(new_tree[0].keys())] = 0
                    else:
                        new_tree = new_tree[vy]
                        batch_masks[i, j, list(new_tree[0].keys())] = 0
                        if new_tree[1] != -1:
                            batch_masks[i, j, list(resettrie[0].keys())] = 0
                p_gen_mask.append(0)
                # batch_masks[i, j, -1] = 0
            p_gen_masks.append(p_gen_mask + [1] * (seqlen - len(p_gen_mask)))
        p_gen_masks = torch.Tensor(p_gen_masks).to(yseqs.device).byte()

        return batch_masks, p_gen_masks

    def get_tcpgen_step(self, vy, trie, resettrie):
        new_tree = trie[0]
        if vy in [self.blank_idx]:
            new_tree = resettrie
        elif self.char_list[vy].endswith("▁"):
            if vy in new_tree and new_tree[vy][0] != {}:
                new_tree = new_tree[vy]
            else:
                new_tree = resettrie
        elif vy not in new_tree:
            new_tree = [{}]
        else:
            new_tree = new_tree[vy]
        return new_tree

    def join(
        self,
        source_encodings: torch.Tensor,
        source_lengths: torch.Tensor,
        target_encodings: torch.Tensor,
        target_lengths: torch.Tensor,
        hptr: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        r"""Applies joint network to source and target encodings.

        B: batch size;
        T: maximum source sequence length in batch;
        U: maximum target sequence length in batch;
        D: dimension of each source and target sequence encoding.
        A: TCPGen attention dimension

        Args:
            source_encodings (torch.Tensor): source encoding sequences, with
                shape `(B, T, D)`.
            source_lengths (torch.Tensor): with shape `(B,)` and i-th element representing
                valid sequence length of i-th batch element in ``source_encodings``.
            target_encodings (torch.Tensor): target encoding sequences, with shape `(B, U, D)`.
            target_lengths (torch.Tensor): with shape `(B,)` and i-th element representing
                valid sequence length of i-th batch element in ``target_encodings``.
            hptr (torch.Tensor): deep biasing vector with shape `(B, T, U, A)`.

        Returns:
            (torch.Tensor, torch.Tensor, torch.Tensor):
                torch.Tensor
                    joint network output, with shape `(B, T, U, output_dim)`.
                torch.Tensor
                    output source lengths, with shape `(B,)` and i-th element representing
                    number of valid elements along dim 1 for i-th batch element in joint network output.
                torch.Tensor
                    joint network second last layer output, with shape `(B, T, U, D)`.
        """
        output, source_lengths, target_lengths, jointer_activation = self.joiner(
            source_encodings=source_encodings,
            source_lengths=source_lengths,
            target_encodings=target_encodings,
            target_lengths=target_lengths,
            hptr=hptr,
        )
        return output, source_lengths, jointer_activation


def conformer_rnnt_model(
    *,
    input_dim: int,
    encoding_dim: int,
    time_reduction_stride: int,
    conformer_input_dim: int,
    conformer_ffn_dim: int,
    conformer_num_layers: int,
    conformer_num_heads: int,
    conformer_depthwise_conv_kernel_size: int,
    conformer_dropout: float,
    num_symbols: int,
    symbol_embedding_dim: int,
    num_lstm_layers: int,
    lstm_hidden_dim: int,
    lstm_layer_norm: int,
    lstm_layer_norm_epsilon: int,
    lstm_dropout: int,
    joiner_activation: str,
) -> RNNT:
    r"""Builds Conformer-based recurrent neural network transducer (RNN-T) model.

    Args:
        input_dim (int): dimension of input sequence frames passed to transcription network.
        encoding_dim (int): dimension of transcription- and prediction-network-generated encodings
            passed to joint network.
        time_reduction_stride (int): factor by which to reduce length of input sequence.
        conformer_input_dim (int): dimension of Conformer input.
        conformer_ffn_dim (int): hidden layer dimension of each Conformer layer's feedforward network.
        conformer_num_layers (int): number of Conformer layers to instantiate.
        conformer_num_heads (int): number of attention heads in each Conformer layer.
        conformer_depthwise_conv_kernel_size (int): kernel size of each Conformer layer's depthwise convolution layer.
        conformer_dropout (float): Conformer dropout probability.
        num_symbols (int): cardinality of set of target tokens.
        symbol_embedding_dim (int): dimension of each target token embedding.
        num_lstm_layers (int): number of LSTM layers to instantiate.
        lstm_hidden_dim (int): output dimension of each LSTM layer.
        lstm_layer_norm (bool): if ``True``, enables layer normalization for LSTM layers.
        lstm_layer_norm_epsilon (float): value of epsilon to use in LSTM layer normalization layers.
        lstm_dropout (float): LSTM dropout probability.
        joiner_activation (str): activation function to use in the joiner.
            Must be one of ("relu", "tanh"). (Default: "relu")

        Returns:
            RNNT:
                Conformer RNN-T model.
    """
    encoder = _ConformerEncoder(
        input_dim=input_dim,
        output_dim=encoding_dim,
        time_reduction_stride=time_reduction_stride,
        conformer_input_dim=conformer_input_dim,
        conformer_ffn_dim=conformer_ffn_dim,
        conformer_num_layers=conformer_num_layers,
        conformer_num_heads=conformer_num_heads,
        conformer_depthwise_conv_kernel_size=conformer_depthwise_conv_kernel_size,
        conformer_dropout=conformer_dropout,
    )
    predictor = _Predictor(
        num_symbols=num_symbols,
        output_dim=encoding_dim,
        symbol_embedding_dim=symbol_embedding_dim,
        num_lstm_layers=num_lstm_layers,
        lstm_hidden_dim=lstm_hidden_dim,
        lstm_layer_norm=lstm_layer_norm,
        lstm_layer_norm_epsilon=lstm_layer_norm_epsilon,
        lstm_dropout=lstm_dropout,
    )
    joiner = _Joiner(encoding_dim, num_symbols, activation=joiner_activation)
    return RNNT(encoder, predictor, joiner)



def conformer_rnnt_base() -> RNNT:
    r"""Builds basic version of Conformer RNN-T model.

    Returns:
        RNNT:
            Conformer RNN-T model.
    """
    return conformer_rnnt_model(
        input_dim=80,
        encoding_dim=1024,
        time_reduction_stride=4,
        conformer_input_dim=256,
        conformer_ffn_dim=1024,
        conformer_num_layers=1, #Originally it was 16
        conformer_num_heads=4,
        conformer_depthwise_conv_kernel_size=31,
        conformer_dropout=0.1,
        num_symbols=1024,
        symbol_embedding_dim=256,
        num_lstm_layers=2,
        lstm_hidden_dim=512,
        lstm_layer_norm=True,
        lstm_layer_norm_epsilon=1e-5,
        lstm_dropout=0.3,
        joiner_activation="tanh",
    )



def conformer_rnnt_biasing(
    *,
    input_dim: int,
    encoding_dim: int,
    time_reduction_stride: int,
    conformer_input_dim: int,
    conformer_ffn_dim: int,
    conformer_num_layers: int,
    conformer_num_heads: int,
    conformer_depthwise_conv_kernel_size: int,
    conformer_dropout: float,
    num_symbols: int,
    symbol_embedding_dim: int,
    num_lstm_layers: int,
    lstm_hidden_dim: int,
    lstm_layer_norm: int,
    lstm_layer_norm_epsilon: int,
    lstm_dropout: int,
    joiner_activation: str,
    attndim: int,
    biasing: bool,
    charlist: List[str],
    deepbiasing: bool,
    tcpsche: int,
    DBaverage: bool,
) -> RNNTBiasing:
    r"""Builds Conformer-based recurrent neural network transducer (RNN-T) model.

    Args:
        input_dim (int): dimension of input sequence frames passed to transcription network.
        encoding_dim (int): dimension of transcription- and prediction-network-generated encodings
            passed to joint network.
        time_reduction_stride (int): factor by which to reduce length of input sequence.
        conformer_input_dim (int): dimension of Conformer input.
        conformer_ffn_dim (int): hidden layer dimension of each Conformer layer's feedforward network.
        conformer_num_layers (int): number of Conformer layers to instantiate.
        conformer_num_heads (int): number of attention heads in each Conformer layer.
        conformer_depthwise_conv_kernel_size (int): kernel size of each Conformer layer's depthwise convolution layer.
        conformer_dropout (float): Conformer dropout probability.
        num_symbols (int): cardinality of set of target tokens.
        symbol_embedding_dim (int): dimension of each target token embedding.
        num_lstm_layers (int): number of LSTM layers to instantiate.
        lstm_hidden_dim (int): output dimension of each LSTM layer.
        lstm_layer_norm (bool): if ``True``, enables layer normalization for LSTM layers.
        lstm_layer_norm_epsilon (float): value of epsilon to use in LSTM layer normalization layers.
        lstm_dropout (float): LSTM dropout probability.
        joiner_activation (str): activation function to use in the joiner.
            Must be one of ("relu", "tanh"). (Default: "relu")
        attndim (int): TCPGen attention dimension
        biasing (bool): If true, use biasing, otherwise use standard RNN-T
        charlist (list): The list of word piece tokens in the same order as the output layer
        deepbiasing (bool): If true, use deep biasing by extracting the biasing vector
        tcpsche (int): The epoch at which TCPGen starts to train
        DBaverage (bool): If true, instead of TCPGen, use DBRNNT for biasing

        Returns:
            RNNT:
                Conformer RNN-T model with TCPGen-based biasing support.
    """
    encoder = _ConformerEncoder(
        input_dim=input_dim,
        output_dim=encoding_dim,
        time_reduction_stride=time_reduction_stride,
        conformer_input_dim=conformer_input_dim,
        conformer_ffn_dim=conformer_ffn_dim,
        conformer_num_layers=conformer_num_layers,
        conformer_num_heads=conformer_num_heads,
        conformer_depthwise_conv_kernel_size=conformer_depthwise_conv_kernel_size,
        conformer_dropout=conformer_dropout,
    )
    predictor = _Predictor(
        num_symbols=num_symbols,
        output_dim=encoding_dim,
        symbol_embedding_dim=symbol_embedding_dim,
        num_lstm_layers=num_lstm_layers,
        lstm_hidden_dim=lstm_hidden_dim,
        lstm_layer_norm=lstm_layer_norm,
        lstm_layer_norm_epsilon=lstm_layer_norm_epsilon,
        lstm_dropout=lstm_dropout,
    )
    joiner = _JoinerBiasing(
        encoding_dim,
        num_symbols,
        activation=joiner_activation,
        deepbiasing=deepbiasing,
        attndim=attndim,
        biasing=biasing,
    )
    return RNNTBiasing(
        encoder,
        predictor,
        joiner,
        attndim,
        biasing,
        deepbiasing,
        symbol_embedding_dim,
        encoding_dim,
        charlist,
        encoding_dim,
        conformer_dropout,
        tcpsche,
        DBaverage,
    )


def conformer_rnnt_biasing_base(charlist=None, biasing=True) -> RNNT:
    r"""Builds basic version of Conformer RNN-T model with TCPGen.

    Returns:
        RNNT:
            Conformer RNN-T model with TCPGen-based biasing support.
    """
    return conformer_rnnt_biasing(
        input_dim=80,
        encoding_dim=576,
        time_reduction_stride=4,
        conformer_input_dim=144,
        conformer_ffn_dim=576,
        conformer_num_layers=16,
        conformer_num_heads=4,
        conformer_depthwise_conv_kernel_size=31,
        conformer_dropout=0.1,
        num_symbols=601,
        symbol_embedding_dim=256,
        num_lstm_layers=1,
        lstm_hidden_dim=320,
        lstm_layer_norm=True,
        lstm_layer_norm_epsilon=1e-5,
        lstm_dropout=0.3,
        joiner_activation="tanh",
        attndim=256,
        biasing=biasing,
        charlist=charlist,
        deepbiasing=True,
        tcpsche=30,
        DBaverage=False,
    )


__all__ = ["Hypothesis", "RNNTBeamSearch"]


Hypothesis = Tuple[List[int], torch.Tensor, List[List[torch.Tensor]], float]
Hypothesis.__doc__ = """Hypothesis generated by RNN-T beam search decoder,
    represented as tuple of (tokens, prediction network output, prediction network state, score).
    """


def _get_hypo_tokens(hypo: Hypothesis) -> List[int]:
    return hypo[0]


def _get_hypo_predictor_out(hypo: Hypothesis) -> torch.Tensor:
    return hypo[1]


def _get_hypo_state(hypo: Hypothesis) -> List[List[torch.Tensor]]:
    return hypo[2]


def _get_hypo_score(hypo: Hypothesis) -> float:
    return hypo[3]


def _get_hypo_key(hypo: Hypothesis) -> str:
    return str(hypo[0])


def _batch_state(hypos: List[Hypothesis]) -> List[List[torch.Tensor]]:
    states: List[List[torch.Tensor]] = []
    for i in range(len(_get_hypo_state(hypos[0]))):
        batched_state_components: List[torch.Tensor] = []
        for j in range(len(_get_hypo_state(hypos[0])[i])):
            batched_state_components.append(torch.cat([_get_hypo_state(hypo)[i][j] for hypo in hypos]))
        states.append(batched_state_components)
    return states


def _slice_state(states: List[List[torch.Tensor]], idx: int, device: torch.device) -> List[List[torch.Tensor]]:
    idx_tensor = torch.tensor([idx], device=device)
    return [[state.index_select(0, idx_tensor) for state in state_tuple] for state_tuple in states]


def _default_hypo_sort_key(hypo: Hypothesis) -> float:
    return _get_hypo_score(hypo) / (len(_get_hypo_tokens(hypo)) + 1)


def _compute_updated_scores(
    hypos: List[Hypothesis],
    next_token_probs: torch.Tensor,
    beam_width: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    hypo_scores = torch.tensor([_get_hypo_score(h) for h in hypos]).unsqueeze(1)
    nonblank_scores = hypo_scores + next_token_probs[:, :-1]  # [beam_width, num_tokens - 1]
    nonblank_nbest_scores, nonblank_nbest_idx = nonblank_scores.reshape(-1).topk(beam_width)
    nonblank_nbest_hypo_idx = nonblank_nbest_idx.div(nonblank_scores.shape[1], rounding_mode="trunc")
    nonblank_nbest_token = nonblank_nbest_idx % nonblank_scores.shape[1]
    return nonblank_nbest_scores, nonblank_nbest_hypo_idx, nonblank_nbest_token


def _remove_hypo(hypo: Hypothesis, hypo_list: List[Hypothesis]) -> None:
    for i, elem in enumerate(hypo_list):
        if _get_hypo_key(hypo) == _get_hypo_key(elem):
            del hypo_list[i]
            break


class RNNTBeamSearch(torch.nn.Module):
    r"""Beam search decoder for RNN-T model.

    See Also:
        * :class:`torchaudio.pipelines.RNNTBundle`: ASR pipeline with pretrained model.

    Args:
        model (RNNT): RNN-T model to use.
        blank (int): index of blank token in vocabulary.
        temperature (float, optional): temperature to apply to joint network output.
            Larger values yield more uniform samples. (Default: 1.0)
        hypo_sort_key (Callable[[Hypothesis], float] or None, optional): callable that computes a score
            for a given hypothesis to rank hypotheses by. If ``None``, defaults to callable that returns
            hypothesis score normalized by token sequence length. (Default: None)
        step_max_tokens (int, optional): maximum number of tokens to emit per input time step. (Default: 100)
    """

    def __init__(
        self,
        model: RNNT,
        blank: int,
        temperature: float = 1.0,
        hypo_sort_key: Optional[Callable[[Hypothesis], float]] = None,
        step_max_tokens: int = 100,
    ) -> None:
        super().__init__()
        self.model = model
        self.blank = blank
        self.temperature = temperature

        if hypo_sort_key is None:
            self.hypo_sort_key = _default_hypo_sort_key
        else:
            self.hypo_sort_key = hypo_sort_key

        self.step_max_tokens = step_max_tokens

    def _init_b_hypos(self, device: torch.device) -> List[Hypothesis]:
        token = self.blank
        state = None

        one_tensor = torch.tensor([1], device=device)
        pred_out, _, pred_state = self.model.predict(torch.tensor([[token]], device=device), one_tensor, state)
        init_hypo = (
            [token],
            pred_out[0].detach(),
            pred_state,
            0.0,
        )
        return [init_hypo]

    def _gen_next_token_probs(
        self, enc_out: torch.Tensor, hypos: List[Hypothesis], device: torch.device
    ) -> torch.Tensor:
        one_tensor = torch.tensor([1], device=device)
        predictor_out = torch.stack([_get_hypo_predictor_out(h) for h in hypos], dim=0)
        joined_out, _, _ = self.model.join(
            enc_out,
            one_tensor,
            predictor_out,
            torch.tensor([1] * len(hypos), device=device),
        )  # [beam_width, 1, 1, num_tokens]
        joined_out = torch.nn.functional.log_softmax(joined_out / self.temperature, dim=3)
        return joined_out[:, 0, 0]

    def _gen_b_hypos(
        self,
        b_hypos: List[Hypothesis],
        a_hypos: List[Hypothesis],
        next_token_probs: torch.Tensor,
        key_to_b_hypo: Dict[str, Hypothesis],
    ) -> List[Hypothesis]:
        for i in range(len(a_hypos)):
            h_a = a_hypos[i]
            append_blank_score = _get_hypo_score(h_a) + next_token_probs[i, -1]
            if _get_hypo_key(h_a) in key_to_b_hypo:
                h_b = key_to_b_hypo[_get_hypo_key(h_a)]
                _remove_hypo(h_b, b_hypos)
                score = float(torch.tensor(_get_hypo_score(h_b)).logaddexp(append_blank_score))
            else:
                score = float(append_blank_score)
            h_b = (
                _get_hypo_tokens(h_a),
                _get_hypo_predictor_out(h_a),
                _get_hypo_state(h_a),
                score,
            )
            b_hypos.append(h_b)
            key_to_b_hypo[_get_hypo_key(h_b)] = h_b
        _, sorted_idx = torch.tensor([_get_hypo_score(hypo) for hypo in b_hypos]).sort()
        return [b_hypos[idx] for idx in sorted_idx]

    def _gen_a_hypos(
        self,
        a_hypos: List[Hypothesis],
        b_hypos: List[Hypothesis],
        next_token_probs: torch.Tensor,
        t: int,
        beam_width: int,
        device: torch.device,
    ) -> List[Hypothesis]:
        (
            nonblank_nbest_scores,
            nonblank_nbest_hypo_idx,
            nonblank_nbest_token,
        ) = _compute_updated_scores(a_hypos, next_token_probs, beam_width)

        if len(b_hypos) < beam_width:
            b_nbest_score = -float("inf")
        else:
            b_nbest_score = _get_hypo_score(b_hypos[-beam_width])

        base_hypos: List[Hypothesis] = []
        new_tokens: List[int] = []
        new_scores: List[float] = []
        for i in range(beam_width):
            score = float(nonblank_nbest_scores[i])
            if score > b_nbest_score:
                a_hypo_idx = int(nonblank_nbest_hypo_idx[i])
                base_hypos.append(a_hypos[a_hypo_idx])
                new_tokens.append(int(nonblank_nbest_token[i]))
                new_scores.append(score)

        if base_hypos:
            new_hypos = self._gen_new_hypos(base_hypos, new_tokens, new_scores, t, device)
        else:
            new_hypos: List[Hypothesis] = []

        return new_hypos

    def _gen_new_hypos(
        self,
        base_hypos: List[Hypothesis],
        tokens: List[int],
        scores: List[float],
        t: int,
        device: torch.device,
    ) -> List[Hypothesis]:
        tgt_tokens = torch.tensor([[token] for token in tokens], device=device)
        states = _batch_state(base_hypos)
        pred_out, _, pred_states = self.model.predict(
            tgt_tokens,
            torch.tensor([1] * len(base_hypos), device=device),
            states,
        )
        new_hypos: List[Hypothesis] = []
        for i, h_a in enumerate(base_hypos):
            new_tokens = _get_hypo_tokens(h_a) + [tokens[i]]
            new_hypos.append((new_tokens, pred_out[i].detach(), _slice_state(pred_states, i, device), scores[i]))
        return new_hypos

    def _search(
        self,
        enc_out: torch.Tensor,
        hypo: Optional[List[Hypothesis]],
        beam_width: int,
    ) -> List[Hypothesis]:
        n_time_steps = enc_out.shape[1]
        device = enc_out.device

        a_hypos: List[Hypothesis] = []
        b_hypos = self._init_b_hypos(device) if hypo is None else hypo
        for t in range(n_time_steps):
            a_hypos = b_hypos
            b_hypos = torch.jit.annotate(List[Hypothesis], [])
            key_to_b_hypo: Dict[str, Hypothesis] = {}
            symbols_current_t = 0

            while a_hypos:
                next_token_probs = self._gen_next_token_probs(enc_out[:, t : t + 1], a_hypos, device)
                next_token_probs = next_token_probs.cpu()
                b_hypos = self._gen_b_hypos(b_hypos, a_hypos, next_token_probs, key_to_b_hypo)

                if symbols_current_t == self.step_max_tokens:
                    break

                a_hypos = self._gen_a_hypos(
                    a_hypos,
                    b_hypos,
                    next_token_probs,
                    t,
                    beam_width,
                    device,
                )
                if a_hypos:
                    symbols_current_t += 1

            _, sorted_idx = torch.tensor([self.hypo_sort_key(hyp) for hyp in b_hypos]).topk(beam_width)
            b_hypos = [b_hypos[idx] for idx in sorted_idx]

        return b_hypos

    def forward(self, input: torch.Tensor, length: torch.Tensor, beam_width: int) -> List[Hypothesis]:
        r"""Performs beam search for the given input sequence.

        T: number of frames;
        D: feature dimension of each frame.

        Args:
            input (torch.Tensor): sequence of input frames, with shape (T, D) or (1, T, D).
            length (torch.Tensor): number of valid frames in input
                sequence, with shape () or (1,).
            beam_width (int): beam size to use during search.

        Returns:
            List[Hypothesis]: top-``beam_width`` hypotheses found by beam search.
        """
        if input.dim() != 2 and not (input.dim() == 3 and input.shape[0] == 1):
            raise ValueError("input must be of shape (T, D) or (1, T, D)")
        if input.dim() == 2:
            input = input.unsqueeze(0)

        if length.shape != () and length.shape != (1,):
            raise ValueError("length must be of shape () or (1,)")
        if length.dim() == 0:
            length = length.unsqueeze(0)

        enc_out, _ = self.model.transcribe(input, length)
        return self._search(enc_out, None, beam_width)


    @torch.jit.export
    def infer(
        self,
        input: torch.Tensor,
        length: torch.Tensor,
        beam_width: int,
        state: Optional[List[List[torch.Tensor]]] = None,
        hypothesis: Optional[List[Hypothesis]] = None,
    ) -> Tuple[List[Hypothesis], List[List[torch.Tensor]]]:
        r"""Performs beam search for the given input sequence in streaming mode.

        T: number of frames;
        D: feature dimension of each frame.

        Args:
            input (torch.Tensor): sequence of input frames, with shape (T, D) or (1, T, D).
            length (torch.Tensor): number of valid frames in input
                sequence, with shape () or (1,).
            beam_width (int): beam size to use during search.
            state (List[List[torch.Tensor]] or None, optional): list of lists of tensors
                representing transcription network internal state generated in preceding
                invocation. (Default: ``None``)
            hypothesis (List[Hypothesis] or None): hypotheses from preceding invocation to seed
                search with. (Default: ``None``)

        Returns:
            (List[Hypothesis], List[List[torch.Tensor]]):
                List[Hypothesis]
                    top-``beam_width`` hypotheses found by beam search.
                List[List[torch.Tensor]]
                    list of lists of tensors representing transcription network
                    internal state generated in current invocation.
        """
        if input.dim() != 2 and not (input.dim() == 3 and input.shape[0] == 1):
            raise ValueError("input must be of shape (T, D) or (1, T, D)")
        if input.dim() == 2:
            input = input.unsqueeze(0)

        if length.shape != () and length.shape != (1,):
            raise ValueError("length must be of shape () or (1,)")
        if length.dim() == 0:
            length = length.unsqueeze(0)

        enc_out, _, state = self.model.transcribe_streaming(input, length, state)
        return self._search(enc_out, hypothesis, beam_width), state