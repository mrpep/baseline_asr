import os

os.environ['HF_DATASETS_OFFLINE']='1'
os.environ['HF_EVALUATE_OFFLINE']='1'

from datasets import load_dataset, load_metric
from utils import preprocess_timit_transcriptions, create_vocab, prepare_timit, DataCollatorCTCWithPadding, compute_metrics
from transformers import Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor, Wav2Vec2Processor,Wav2Vec2ForCTC, TrainingArguments, Trainer, WavLMForCTC
import datasets
import evaluate
from safe_gpu import safe_gpu
import logging
from functools import partial
from pathlib import Path

logging.basicConfig(level=logging.INFO)
safe_gpu.claim_gpus()
logging.info(f'Allocated devices: {safe_gpu.gpu_owner.devices_taken}')

logging.info('Loading dataset')
timit_out_path = '/tmp/qpepino/hf_data/timit-hf'

timit = datasets.load_from_disk(timit_out_path)
logging.info('Processing transcripts')
timit = preprocess_timit_transcriptions(timit)
if not Path('./vocab.json').exists():
  create_vocab(timit)
else:
  logging.info('Reusing existing vocab.json')

tokenizer = Wav2Vec2CTCTokenizer("./vocab.json", unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|")
logging.info('Preparing dataset')
feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True, return_attention_mask=True)
processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)
timit = prepare_timit(timit, processor, num_proc=4)

logging.info('Data collator')
data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)

logging.info('Loading WER')
wer_metric = evaluate.load("wer")

logging.info('Instantiating model')
#model = Wav2Vec2ForCTC.from_pretrained(
#    "facebook/wav2vec2-base", 
#    ctc_loss_reduction="mean", 
#    pad_token_id=processor.tokenizer.pad_token_id,
#    local_files_only=True
#)
model = WavLMForCTC.from_pretrained(
  "microsoft/wavlm-base-plus",
  ctc_loss_reduction="mean",
  pad_token_id=processor.tokenizer.pad_token_id,
  local_files_only=True
)
model.freeze_feature_extractor()

training_args = TrainingArguments(
  output_dir='baseline-timit-wavlmplus-fixmask',
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
    compute_metrics=partial(compute_metrics,wer_metric=wer_metric, processor=processor),
    train_dataset=timit["train"],
    eval_dataset=timit["test"],
    tokenizer=processor.feature_extractor
)
logging.info('Training')
trainer.train()





