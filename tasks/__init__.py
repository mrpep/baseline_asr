from pathlib import Path
from loguru import logger
import re
import json
from functools import partial
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
import torch
import numpy as np
from transformers import AutoProcessor, Trainer
from tqdm import tqdm
import joblib

def global_setup(state):
    os.environ['HF_DATASETS_OFFLINE']='1'
    os.environ['HF_EVALUATE_OFFLINE']='1'
    os.environ['WANDB_DISABLED'] = 'True'

    from safe_gpu import safe_gpu
    
    safe_gpu.claim_gpus()
    logger.info(f'Allocated devices: {safe_gpu.gpu_owner.devices_taken}')

    state['allocated_devices'] = safe_gpu.gpu_owner.devices_taken
    return state

def load_hf_dataset(state, dataset_fn, save_path, postprocessing_fn=None):
    import datasets
    if Path(save_path).exists():
        logger.info('There is already a dataset in {}. Loading it...'.format(save_path))
        dataset = datasets.load_from_disk(save_path)
    else:
        dataset = dataset_fn()
        dataset.save_to_disk(save_path)

    if postprocessing_fn is not None:
        for fn in postprocessing_fn:
            state, dataset = fn(state, dataset)

    state['dataset'] = dataset
    return state

def preprocess_transcriptions(state, dataset, chars_to_ignore_regex='[\,\?\.\!\-\;\:\"]', remove_columns=["phonetic_detail", "word_detail", "dialect_region", "id", "sentence_type", "speaker_id"]):
    def remove_special_characters(batch):
        batch["text"] = re.sub(chars_to_ignore_regex, '', batch["text"]).lower()
        return batch
    
    dataset = dataset.map(remove_special_characters)
    dataset = dataset.remove_columns(remove_columns)

    return state, dataset

def extract_all_chars(batch):
  all_text = " ".join(batch["text"])
  vocab = list(set(all_text))
  return {"vocab": [vocab], "all_text": [all_text]}

def create_char_vocab(state, dataset, vocab_path):
    timit = dataset
    if Path(vocab_path).exists():
        logger.info('Reusing existing vocab: {}'.format(vocab_path))
    else:
        vocabs = timit.map(extract_all_chars, batched=True, batch_size=-1, keep_in_memory=True, remove_columns=timit.column_names["train"])
        vocab_list = list(set(vocabs["train"]["vocab"][0]) | set(vocabs["test"]["vocab"][0]))
        vocab_dict = {v: k for k, v in enumerate(vocab_list)}
        vocab_dict["|"] = vocab_dict[" "]
        del vocab_dict[" "]
        vocab_dict["[UNK]"] = len(vocab_dict)
        vocab_dict["[PAD]"] = len(vocab_dict)
        if not Path(vocab_path).parent.exists():
            Path(vocab_path).parent.mkdir(parents=True)
        with open(vocab_path, 'w') as vocab_file:
            json.dump(vocab_dict, vocab_file)
    state['vocab_path'] = vocab_path
    return state, dataset

def load_hf_model(state, model_cls, from_pretrained_args=None, freeze_feature_extractor=True):
    if from_pretrained_args is None:
        from_pretrained_args = {}
    from_pretrained_args['pad_token_id'] = state['processor'].tokenizer.pad_token_id
    from_pretrained_args['vocab_size'] = len(state['processor'].tokenizer)
    model = model_cls.from_pretrained(**from_pretrained_args)
    if freeze_feature_extractor:
        model.freeze_feature_extractor()
    
    state['model'] = model
    return state
    
def prepare_timit(state, timit, num_proc=4):
    def prepare_dataset(batch, processor):
        audio = batch["audio"]

        # batched output is "un-batched" to ensure mapping is correct
        batch['input_values'] = processor(audio["array"], sampling_rate=audio["sampling_rate"]).input_values[0]
        batch["input_length"] = len(batch["input_values"])

        with processor.as_target_processor():
            batch["labels"] = processor(batch["text"]).input_ids
        return batch

    return state, timit.map(partial(prepare_dataset,processor=state['processor']), num_proc=num_proc)

def load_hf_processor(state, processor, tokenizer=None, feature_extractor=None, from_pretrained=False, local_files_only=True):

    if from_pretrained != False:
        state['processor'] = processor.from_pretrained(from_pretrained, local_files_only=local_files_only)
        logger.info('Using processor from {}'.format(from_pretrained))
    else:
        tokenizer = tokenizer(state['vocab_path'], 
                                unk_token="[UNK]", 
                                pad_token="[PAD]", 
                                word_delimiter_token="|")
        state['processor'] = processor(tokenizer=tokenizer, feature_extractor=feature_extractor())
    return state

def process_hf_dataset(state, processors=None):
    dataset = state['dataset']
    #dataset.pop('train')
    if processors is None:
        processors = []
    for fn in processors:
        state, dataset = fn(state, dataset)
    state['dataset'] = dataset
    return state

def make_dict(**kwargs):
    return dict(**kwargs)

def fit_model(state, training_args, data_collator=None, metrics_fn=None, eval_splits=['test'],trainer_cls=None):
    from transformers import Trainer
    training_args = training_args(output_dir=state['output_dir'])
    if trainer_cls is None:
        trainer_cls = Trainer
    trainer = trainer_cls(
        model=state['model'],
        data_collator=data_collator(state['processor']),
        args=training_args,
        compute_metrics=partial(metrics_fn,processor=state['processor']),
        train_dataset=state['dataset']["train"],
        eval_dataset={si: state['dataset'][si] for si in eval_splits},
        tokenizer=state['processor'].feature_extractor #Is this right? Term mismatch with NLP?
    )
    
    batch = data_collator(state['processor'])([state['dataset']['train'][i] for i in range(16)])
    batch = {k: v.to('cuda') for k,v in batch.items()}
    state['model'].to('cuda')
    outs = state['model'](**batch)
    trainer.train()

class DataCollatorCTCWithPadding:
    def __init__(self, processor,
        padding = True,
        max_length = None,
        max_length_labels = None,
        pad_to_multiple_of = None,
        pad_to_multiple_of_labels = None):

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

def compute_metrics(pred, processor=None):
    import evaluate
    from pyctcdecode import build_ctcdecoder
    from tqdm import tqdm

    wer_metric = evaluate.load('wer')
    pred_logits = pred.predictions
    pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id
    pred_ids = np.argmax(pred_logits, axis=-1)
    label_str = processor.batch_decode(pred.label_ids, group_tokens=False)
    pred_str = processor.batch_decode(pred_ids)
    logger.info("Predictions: {}".format(pred_str[:5]))
    logger.info("GT: {}".format(label_str[:5]))
    wer = wer_metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}

def eval_model(state, beam_size=1, start=0, end=None, split=None):
    import evaluate
    from tasks.models.rnnt import RNNTBeamSearch
    
    base_model = state['model'].wavlm.to('cuda')
    lm_head = state['model'].lm_head.to('cuda')
    blank_token = state['model'].config.vocab_size
    with open(state['vocab_path'],'r') as f:
        vocab = json.load(f)
    idx_to_vocab = {v:k for k,v in vocab.items()}
    state['model'].eval()
    rnnt_decoder = RNNTBeamSearch(lm_head, blank_token)
    results = {}
    for k,v in state['dataset'].items():
        if (split is not None) and (k in split):
            if end is None:
                end = len(v) + 100
            wer_metric = evaluate.load('wer')
            preds = []
            refs = []
            nbests = []
            nbests_scores = []
            ids = []
            for i in tqdm(range(start, end)):
                if i < len(v):
                    x = v[i]
                    with torch.no_grad():
                        outputs = base_model(torch.tensor(x['input_values'], device=base_model.device, dtype=base_model.dtype).unsqueeze(0))
                        outputs = outputs['last_hidden_state']
                        hyp = rnnt_decoder(outputs, torch.tensor([outputs.shape[1]], device=outputs.device), beam_size)
                        scores = [h[-1] for h in hyp]
                        best_hyp = scores.index(max(scores))
                        result = hyp[best_hyp][0]
                        decoded = [idx_to_vocab[xi] for xi in result[1:]]
                        decoded =  ''.join(decoded).replace('|',' ')

                        nbest_i = []
                        nbest_scores_i = []
                        for h in hyp:
                            transc = ''.join([idx_to_vocab[xi] for xi in h[0][1:]]).replace('|',' ')
                            nbest_i.append(transc)
                            nbest_scores_i.append(h[-1])
                        ordered_scores_idxs = np.argsort(nbest_scores_i)[::-1]
                        nbest_i = np.array(nbest_i)[ordered_scores_idxs]
                        nbest_scores_i = np.array(nbest_scores_i)[ordered_scores_idxs]

                        preds.append(decoded)
                        nbests.append(nbest_i)
                        nbests_scores.append(nbest_scores_i)
                        ref = ''.join([idx_to_vocab[xi] for xi in x['labels']]).replace('|',' ')
                        refs.append(ref)
                        ids.append(x['id'])

            wer = wer_metric.compute(predictions=preds, references=refs)
            logger.info('{}_wer: {:.3f}'.format(k, wer))
            result_i = {'wer': wer, 'preds': preds, 'refs': refs, 'nbest': nbests, 'nbest_scores': nbests_scores, 'ids': ids}
            joblib.dump(result_i, Path(state['output_dir'],'{}-{}-{}.pkl'.format(k, start, end)))
            results[k] = result_i
    state['results'] = results

    joblib.dump(results, Path(state['output_dir'],'results.pkl'))
    return state
            
    
    
