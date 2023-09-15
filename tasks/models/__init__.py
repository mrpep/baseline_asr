from transformers import WavLMPreTrainedModel, WavLMModel
from torchaudio.functional import rnnt_loss
from torchaudio.models.rnnt import _Predictor, _Joiner, RNNT
import warnings
from transformers.modeling_outputs import (
    BaseModelOutput,
    CausalLMOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
    Wav2Vec2BaseModelOutput,
    XVectorOutput,
)
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import add_code_sample_docstrings, add_start_docstrings, add_start_docstrings_to_model_forward, logging
from tasks.models.rnnt import conformer_rnnt_model, RNNTBeamSearch
from transformers import Trainer
from loguru import logger
from tqdm import tqdm
import inspect
import torch.nn as nn
import torch
import transformers

class WavLMRNNT(WavLMPreTrainedModel):
    def __init__(self, config, 
                        target_lang = None, 
                        upstream_model_cls=WavLMModel, 
                        speaker_embedding_dim=None,
                        speaker_embedding_mode='sum',
                        xch_layers=None,
                        pool_layer=None,
                        num_conformer_layers=1):
        super().__init__(config)

        if speaker_embedding_dim is not None:
            config.speaker_embedding_dim = speaker_embedding_dim
            config.speaker_embedding_mode = speaker_embedding_mode
        if xch_layers is not None:
            config.xch_layers = xch_layers
        else:
            config.xch_layers = []
        if pool_layer is not None:
            config.pool_layer = pool_layer
        else:
            config.pool_layer = None
        config.num_conformer_layers=num_conformer_layers
        if 'Multichannel' in upstream_model_cls.__name__:
            config.layerdrop=0
        self.wavlm = upstream_model_cls(config)
        self.dropout = nn.Dropout(config.final_dropout)

        self.target_lang = target_lang

        if config.vocab_size is None:
            raise ValueError(
                f"You are trying to instantiate {self.__class__} with a configuration that "
                "does not define the vocabulary size of the language model head. Please "
                "instantiate the model as follows: `WavLMForCTC.from_pretrained(..., vocab_size=vocab_size)`. "
                "or define `vocab_size` of your model's configuration."
            )
        output_hidden_size = (
            config.output_hidden_size if hasattr(config, "add_adapter") and config.add_adapter else config.hidden_size
        )
        #self.lm_head = nn.Linear(output_hidden_size, config.vocab_size)
        #predictor = _Predictor(config.vocab_size,
        #                       output_hidden_size,
        #                       128,
        #                       2,
        #                       128)
        #joiner = _Joiner(output_hidden_size, config.vocab_size)
        #self.lm_head = RNNT(predictor, joiner)
        self.lm_head = conformer_rnnt_model(
            input_dim=output_hidden_size,
            encoding_dim=output_hidden_size,
            time_reduction_stride=4,
            conformer_input_dim=256,
            conformer_ffn_dim=1024,
            conformer_num_layers=config.num_conformer_layers,
            conformer_num_heads=4,
            conformer_depthwise_conv_kernel_size=31,
            conformer_dropout=0.1,
            num_symbols=config.vocab_size+1, #Add [SOS] token
            symbol_embedding_dim=256,
            num_lstm_layers=2,
            lstm_hidden_dim=512,
            lstm_layer_norm=True,
            lstm_layer_norm_epsilon=1e-5,
            lstm_dropout=0.3,
            joiner_activation='tanh'
        )
        # Initialize weights and apply final processing
        self.post_init()

    def tie_weights(self):
        """
        This method overwrites [`~PreTrainedModel.tie_weights`] so that adapter weights can be correctly loaded when
        passing `target_lang=...` to `from_pretrained(...)`.

        This method is **not** supposed to be called by the user and is prone to be changed in the future.
        """

        # Note that `tie_weights` is usually used to tie input and output embedding weights. The method is re-purposed to
        # correctly load adapter layers for WavLM so that we do not have to introduce a new API to
        # [`PreTrainedModel`]. While slightly hacky, WavLM never has to tie input and output embeddings, so that it is
        # ok to repurpose this function here.
        target_lang = self.target_lang

        if target_lang is not None and getattr(self.config, "adapter_attn_dim", None) is None:
            raise ValueError(f"Cannot pass `target_lang`: {target_lang} if `config.adapter_attn_dim` is not defined.")
        elif target_lang is None and getattr(self.config, "adapter_attn_dim", None) is not None:
            logger.info("By default `target_lang` is set to 'eng'.")
        elif target_lang is not None:
            self.load_adapter(target_lang, force_load=True)

    def freeze_feature_extractor(self):
        """
        Calling this function will disable the gradient computation for the feature encoder so that its parameter will
        not be updated during training.
        """
        warnings.warn(
            "The method `freeze_feature_extractor` is deprecated and will be removed in Transformers v5."
            "Please use the equivalent `freeze_feature_encoder` method instead.",
            FutureWarning,
        )
        self.freeze_feature_encoder()

    def freeze_feature_encoder(self):
        """
        Calling this function will disable the gradient computation for the feature encoder so that its parameter will
        not be updated during training.
        """
        self.wavlm.feature_extractor._freeze_parameters()

    def freeze_base_model(self):
        """
        Calling this function will disable the gradient computation for the base model so that its parameters will not
        be updated during training. Only the classification head will be updated.
        """
        for param in self.wavlm.parameters():
            param.requires_grad = False

    def forward(
        self,
        input_values,
        spk_emb = None,
        attention_mask = None,
        output_attentions = None,
        output_hidden_states = None,
        return_dict = None,
        labels = None,
        num_channels=None
    ):
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, target_length)`, *optional*):
            Labels for connectionist temporal classification. Note that `target_length` has to be smaller or equal to
            the sequence length of the output logits. Indices are selected in `[-100, 0, ..., config.vocab_size - 1]`.
            All labels set to `-100` are ignored (masked), the loss is only computed for labels in `[0, ...,
            config.vocab_size - 1]`.
        """

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        kwargs = dict(attention_mask=attention_mask,
                        output_attentions=output_attentions,
                        output_hidden_states=output_hidden_states,
                        return_dict=return_dict)

        
        if 'spk_emb' in inspect.signature(self.wavlm.forward).parameters:
            kwargs['spk_emb'] = spk_emb
        if 'ch_lens' in inspect.signature(self.wavlm.forward).parameters:
            kwargs['ch_lens'] = num_channels
        outputs = self.wavlm(
            input_values,
            **kwargs
        )

        hidden_states = outputs[0]
        hidden_states = self.dropout(hidden_states)

        attention_mask = (attention_mask if attention_mask is not None else torch.ones_like(input_values, dtype=torch.long))
        input_lengths = self._get_feat_extract_output_lengths(attention_mask.sum(-1)).to(torch.long)
        if num_channels is not None:
            idxs = torch.cumsum(torch.cat([torch.tensor([0],device=num_channels.device),num_channels[:-1]]), 0)
            attention_mask = attention_mask[idxs]
            input_lengths = input_lengths[idxs]
        labels_len = (labels!=-100).sum(axis=1)
        labels[labels==-100] = 0
        logits, source_lengths, target_lengths, predictor_state = self.lm_head(hidden_states, input_lengths, torch.cat([torch.ones((labels.shape[0],1), device=labels.device, dtype=labels.dtype)*self.config.vocab_size,labels],axis=1), labels_len+1)

        loss = None
        if labels is not None:
            if labels.max() >= self.config.vocab_size:
                raise ValueError(f"Label values must be <= vocab_size: {self.config.vocab_size}")

            loss = rnnt_loss(logits,labels.to(dtype=torch.int32),source_lengths.to(dtype=torch.int32),target_lengths.to(dtype=torch.int32)-1)
            #logger.info(loss)

        if not return_dict:
            output = (logits,) + outputs[_HIDDEN_STATES_START_POSITION:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutput(
            loss=loss, logits=logits, hidden_states=outputs.hidden_states, attentions=outputs.attentions
        )

class CustomTrainer(Trainer):
    def __init__(self, lr_scheduler=None, num_warmup_steps=0, **kwargs):
        super().__init__(**kwargs)
        self.lr_scheduler = lr_scheduler
        self.num_warmup_steps = num_warmup_steps

    def evaluate(self, eval_dataset, ignore_keys=None, metric_key_prefix='eval'):
        import evaluate
        wer_metric = evaluate.load('wer')
        base_model = self.model.wavlm
        lm_head = self.model.lm_head
        blank_token = self.model.config.vocab_size
        #sos_token = self.model.config.vocab_size
        idx_to_vocab = {v:k for k,v in self.compute_metrics.keywords['processor'].tokenizer.vocab.items()}
        preds = []
        refs = []
        self.model.eval()
        rnnt_decoder = RNNTBeamSearch(self.model.lm_head, blank_token)
        for x in tqdm(eval_dataset):
            with torch.no_grad():
                model_args = [torch.tensor(x['input_values'], device=base_model.device, dtype=base_model.dtype).unsqueeze(0)]
                model_kwargs = {}
                if 'spk_emb' in x:
                    model_kwargs['spk_emb'] = torch.tensor(x['spk_emb'], device=base_model.device, dtype=base_model.dtype).unsqueeze(0)
                if 'num_channels' in x:
                    model_kwargs['ch_lens'] = torch.tensor(x['num_channels'], device=base_model.device, dtype=torch.int32).unsqueeze(0)
                    model_args = [model_args[0][0]]

                outputs = base_model(*model_args, **model_kwargs)
                outputs = outputs['last_hidden_state']
                hyp = rnnt_decoder(outputs, torch.tensor([outputs.shape[1]], device=outputs.device), 1)
                result = hyp[0][0]
                decoded = [idx_to_vocab[xi] for xi in result[1:]]
                decoded =  ''.join(decoded).replace('|',' ')
                preds.append(decoded)
                ref = ''.join([idx_to_vocab[xi] for xi in x['labels']]).replace('|',' ')
                refs.append(ref)

        wer = wer_metric.compute(predictions=preds, references=refs)
        logger.info('Predictions: {}'.format(preds[:5]))
        logger.info('GT: {}'.format(refs[:5]))
        logger.info('{}_WER: {:.3f}'.format(metric_key_prefix,wer))
        self.model.train()
        return {metric_key_prefix+'_wer':wer}
