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
import torch
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
    logging.info('Creating vocab')
    create_vocab(timit)
else:
    logging.info('Caching vocab')

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
#model = Wav2Vec2ForCTC.from_pretrained('baseline-timit-wavlmplus-fixmask/checkpoint-4000')
model = WavLMForCTC.from_pretrained('baseline-timit-wavlmplus-fixmask/checkpoint-4000')
model.freeze_feature_extractor()
model.to('cuda')

def map_to_result(batch):
  with torch.no_grad():
    input_values = torch.tensor(batch["input_values"], device="cuda").unsqueeze(0)
    logits = model(input_values).logits

  pred_ids = torch.argmax(logits, dim=-1)
  batch["pred_str"] = processor.batch_decode(pred_ids)[0]
  batch["text"] = processor.decode(batch["labels"], group_tokens=False)
  
  return batch

results = timit["test"].map(map_to_result, remove_columns=timit["test"].column_names)

logging.info("Test WER: {:.3f}".format(wer_metric.compute(predictions=results["pred_str"], references=results["text"])))


  





