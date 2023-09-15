import os
import numpy as np
os.environ['HF_DATASETS_OFFLINE']='1'
os.environ['HF_EVALUATE_OFFLINE']='1'
os.environ['WANDB_DISABLED'] = 'True'

import datasets
import evaluate
from datasets import load_dataset, load_metric
import re
from IPython import embed
from transformers import Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor, Wav2Vec2Processor,Wav2Vec2ForCTC, TrainingArguments, Trainer, WavLMForCTC
import json
from w2v2_feature_extractor import feature_extractor
import torch
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
#from training_args import trainer

from safe_gpu import safe_gpu
from loguru import logger

def remove_special_characters(batch):
    batch["text"] = re.sub(chars_to_ignore_regex, '', batch["text"]).lower()
    return batch

def extract_all_chars(batch):
  all_text = " ".join(batch["text"])
  vocab = list(set(all_text))
  return {"vocab": [vocab], "all_text": [all_text]}

def prepare_dataset(batch):
    
    audio = batch["audio"]

    # batched output is "un-batched" to ensure mapping is correct
    batch["input_values"] = processor(audio["array"], sampling_rate=audio["sampling_rate"]).input_values[0]
    
    with processor.as_target_processor():
        batch["labels"] = processor(batch["text"]).input_ids
    return batch

class DataCollatorCTCWithPadding:
    """
    Data collator that will dynamically pad the inputs received.
    Args:
        processor (:class:`~transformers.Wav2Vec2Processor`)
            The processor used for proccessing the data.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.tokenization_utils_base.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:
            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence if provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
        max_length (:obj:`int`, `optional`):
            Maximum length of the ``input_values`` of the returned list and optionally padding length (see above).
        max_length_labels (:obj:`int`, `optional`):
            Maximum length of the ``labels`` returned list and optionally padding length (see above).
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the sequence to a multiple of the provided value.
            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
    """

    def __init__(self, processor: Wav2Vec2Processor,
        padding: Union[bool, str] = True,
        max_length: Optional[int] = None,
        max_length_labels: Optional[int] = None,
        pad_to_multiple_of: Optional[int] = None,
        pad_to_multiple_of_labels: Optional[int] = None):

        self.processor=processor
        self.padding=padding
        self.max_length=max_length
        self.max_length_labels=max_length_labels
        self.pad_to_multiple_of=pad_to_multiple_of
        self.pad_to_multiple_of_labels=pad_to_multiple_of_labels

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need
        # different padding methods
        input_features = [{"input_values": feature["input_values"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        with self.processor.as_target_processor():
            labels_batch = self.processor.pad(
                label_features,
                padding=self.padding,
                max_length=self.max_length_labels,
                pad_to_multiple_of=self.pad_to_multiple_of_labels,
                return_tensors="pt",
            )

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        batch["labels"] = labels

        return batch

def compute_metrics(pred):
    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)

    pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id

    pred_str = processor.batch_decode(pred_ids)
    # we do not want to group tokens when computing the metrics
    label_str = processor.batch_decode(pred.label_ids, group_tokens=False)

    wer = wer_metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}


if __name__=='__main__':

    safe_gpu.claim_gpus()
    logger.info(f'Allocated devices: {safe_gpu.gpu_owner.devices_taken}')

    chime=datasets.load_from_disk("/mnt/matylda5/qpepino/hf_data/chime7")

    dataset_path='/mnt/matylda5/qbarchi/finetune_wav2vec/data'
    
    if not os.path.exists(dataset_path):
        os.mkdir(dataset_path)
        
    #timit.save_to_disk(os.path.join(dataset_path,'timit_dataset'))

    chime = chime.remove_columns(["phonetic_detail", "word_detail", "dialect_region", "id", "sentence_type", "speaker_id"])
    chars_to_ignore_regex = '[\,\?\.\!\-\;\:\"]'
    
    chime = chime.map(remove_special_characters)
    vocabs = chime.map(extract_all_chars, batched=True, batch_size=-1, keep_in_memory=True, remove_columns=timit.column_names["train"])

    vocab_list = list(set(vocabs["train"]["vocab"][0]) | set(vocabs["test"]["vocab"][0]))

    vocab_dict = {v: k for k, v in enumerate(vocab_list)}

    vocab_dict["|"] = vocab_dict[" "]
    del vocab_dict[" "]

    vocab_dict["[UNK]"] = len(vocab_dict)
    vocab_dict["[PAD]"] = len(vocab_dict)

    with open('vocab.json', 'w') as vocab_file:
        json.dump(vocab_dict, vocab_file)
    
    vocab_path='vocab.json'
    
    tokenizer,feature_extractor,processor=feature_extractor(vocab_path)

    chime = chime.map(prepare_dataset, remove_columns=chime.column_names["train"], num_proc=4)

    data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)

    #wer_metric = evaluate.load('/mnt/matylda5/qbarchi/finetune_wav2vec/wer.py')
    wer_metric = datasets.load_metric("/mnt/matylda5/qbarchi/finetune_wav2vec/wer")

    # Wav2vec model

#    model = Wav2Vec2ForCTC.from_pretrained(
#    "/mnt/matylda5/qbarchi/finetune_wav2vec/model/wav2vec2-base",
#    ctc_loss_reduction="mean",
#    pad_token_id=processor.tokenizer.pad_token_id,
#    local_files_only=True
#    )
    
    # WavLM model 

    model = WavLMForCTC.from_pretrained('/mnt/matylda5/qbarchi/finetune_wav2vec/model/wavlm/wavlm-base',
                                                # attention_dropout=0.0,
                                                # hidden_dropout=0.0,
                                                # feat_proj_dropout=0.0,
                                                # mask_time_prob=0.05,
                                                # layerdrop=0.0,
                                                ctc_loss_reduction="mean",
                                                pad_token_id=tokenizer.pad_token_id,
                                                vocab_size=len(tokenizer),
                                                )

    model.freeze_feature_extractor()

    training_args = TrainingArguments(
    output_dir=os.path.join('/mnt/matylda5/qbarchi/finetune_wav2vec/checkpoints'),
    group_by_length=True,
    per_device_train_batch_size=32,
    evaluation_strategy="steps",
    num_train_epochs=30,
    fp16=True,
    gradient_checkpointing=True, 
    save_steps=500,
    eval_steps=500,
    logging_steps=500,
    learning_rate=1e-4,
    weight_decay=0.005,
    warmup_steps=1000,
    save_total_limit=2,
    )

    trainer = Trainer(
    model=model,
    data_collator=data_collator,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=chime["train"],
    eval_dataset=chime["test"],
    tokenizer=processor.feature_extractor,
    )

    trainer.train()
    
    if not os.path.exists('checkpoints'):
        os.mkdir('checkpoints')

    #trainer.save_model(os.path.join('checkpoints',f'out_fold{i}'))