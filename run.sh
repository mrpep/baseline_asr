#!/usr/bin/bash
set -x

#$ -N libriaugft
#$ -q long.q@@gpu
#$ -l ram_free=16G,mem_free=16G,gpu=1,gpu_ram=42G
#$ -l matylda5=5
#$ -o /mnt/matylda5/qpepino/baseline-asr/logs/multichannel_asr_4chsel-22-24-lr1e4.o
#$ -e /mnt/matylda5/qpepino/baseline-asr/logs/multichannel_asr_4chsel-22-24-lr1e4.e

# Initialize environment
__conda_setup="$('/mnt/matylda5/qpepino/miniconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)" ; eval "$__conda_setup"
. "/mnt/matylda5/qpepino/miniconda3/etc/profile.d/conda.sh"
unset __conda_setup
unset PYTHONHOME
conda activate multichannel_wavlm

cd /mnt/matylda5/qpepino/baseline-asr

#ginpipe configs/base/train_asr.gin \
#        configs/datasets/chime6_mixer6_dipco_gss_xv.gin \
#        configs/models/wavlm_rnnt_xv.gin \
#        configs/models/training_config.gin \
#        --module_list configs/imports \
#        --project_name chime_asrs \
#        --experiment_name gss-wavlm-large-rnnt-xv-sum \
#        --mods "MODEL_PATH='/mnt/matylda5/qpepino/hf_models/wavlm-large'" "TrainingArguments.eval_steps=10000" "WavLMRNNT.speaker_embedding_mode='sum'" "TrainingArguments.per_device_train_batch_size=8" "TrainingArguments.gradient_accumulation_steps=2"

#ginpipe configs/base/eval_asr.gin \
        #configs/datasets/chime6_mixer6_dipco_gss_deveval.gin \
        #configs/models/wavlm_rnnt.gin \
        #configs/models/training_config.gin \
        #--module_list configs/imports \
       # --project_name chime_asrs \
        #--experiment_name gss-wavlm-large-rnnt-deveval-29500-beam20 \
        #--mods "MODEL_PATH='/mnt/matylda5/qpepino/baseline-asr/experiments/chime_asrs/gss-wavlm-large-rnnt/ckpt29500'" "tasks.eval_model.beam_size=20" "tasks.eval_model.start=22000" "tasks.eval_model.end=25000" "tasks.eval_model.split=['chime6_deveval']"

#ginpipe configs/base/train_asr.gin \
#        configs/datasets/librispeech_augmented.gin \
#        configs/models/wavlm_rnnt.gin \
#        configs/models/training_config.gin \
#        --module_list configs/imports \
#        --project_name chime_asrs \
#        --experiment_name gss-wavlm-large-rnnt-encnotfrozen-lr1e5-augmentedls \
#        --mods "MODEL_PATH='/mnt/matylda5/qpepino/hf_models/wavlm-large'" "TrainingArguments.eval_steps=10000" "tasks.load_hf_model.freeze_feature_extractor=False" "TrainingArguments.learning_rate=0.00001" "TrainingArguments.per_device_train_batch_size=4" "TrainingArguments.gradient_accumulation_steps=4"

#ginpipe configs/base/train_asr.gin \
#        configs/datasets/librimix.gin \
#        configs/models/wavlm_rnnt_xv.gin \
#        configs/models/training_config.gin \
#        --module_list configs/imports \
#        --project_name chime_asrs \
#        --experiment_name gss-wavlm-large-rnnt-xv-librimix \
#        --mods "MODEL_PATH='/mnt/matylda5/qpepino/hf_models/wavlm-large'" "TrainingArguments.eval_steps=10000" "WavLMRNNT.speaker_embedding_mode='sum'" "TrainingArguments.per_device_train_batch_size=8" "TrainingArguments.gradient_accumulation_steps=2"

#ginpipe configs/base/train_asr_torch.gin \
#        configs/datasets/librispeech_augmented.gin \
#        configs/models/wavlm_rnnt.gin \
#        configs/models/training_config.gin \
#        --module_list configs/imports \
#        --project_name chime_asrs \
#        --experiment_name gss-wavlm-large-rnnt-libriaug \
#        --mods "MODEL_PATH='/mnt/matylda5/qpepino/hf_models/wavlm-large'" "TrainingArguments.eval_steps=10000" "TrainingArguments.per_device_train_batch_size=8" "TrainingArguments.gradient_accumulation_steps=2" "tasks.load_hf_model.freeze_feature_extractor=False" "TrainingArguments.learning_rate=0.00001"

#ginpipe configs/base/train_asr_torch.gin \
#        configs/datasets/librimix.gin \
#        configs/models/wavlm_rnnt_xv.gin \
#        configs/models/training_config.gin \
#        --module_list configs/imports \
#        --project_name chime_asrs \
#        --experiment_name gss-wavlm-large-rnnt-librimix \
#        --mods "MODEL_PATH='/mnt/matylda5/qpepino/hf_models/wavlm-large'" "TrainingArguments.eval_steps=10000" "WavLMRNNT.speaker_embedding_mode='sum'" "TrainingArguments.per_device_train_batch_size=8" "TrainingArguments.gradient_accumulation_steps=2" "tasks.load_hf_model.freeze_feature_extractor=False" "TrainingArguments.learning_rate=0.00001"

#ginpipe configs/base/eval_asr.gin \
#        configs/datasets/chime6_mixer6_dipco_gss_deveval.gin \
#        configs/models/wavlm_rnnt.gin \
#        configs/models/training_config.gin \
#        --module_list configs/imports \
#        --project_name chime_asrs \
#        --experiment_name eval-librimix-beam16 \
#        --mods "MODEL_PATH='/mnt/matylda5/qpepino/baseline-asr/experiments/chime_asrs/gss-wavlm-large-rnnt-encnotfrozen/checkpoint-130000'" "tasks.eval_model.beam_size=16" "tasks.eval_model.start=10000" "tasks.eval_model.end=15000" "tasks.eval_model.split=['mixer6_deveval']"


#Finetune Librimix in GSS
#ginpipe configs/base/train_asr_torch.gin \
#        configs/datasets/chime6_mixer6_dipco_torch.gin \
#        configs/models/wavlm_rnnt_xv.gin \
#        configs/models/training_config.gin \
#        --module_list configs/imports \
#        --project_name chime_asrs \
#        --experiment_name gss-wavlm-large-rnnt-librimix-300k-finetunegss-lr1e5-warmupcosinedecay \
#        --mods "MODEL_PATH='/mnt/matylda5/qpepino/baseline-asr/experiments/chime_asrs/gss-wavlm-large-rnnt-librimix/checkpoint-300000'" "TrainingArguments.eval_steps=1000" "WavLMRNNT.speaker_embedding_mode='sum'" "TrainingArguments.per_device_train_batch_size=8" "TrainingArguments.gradient_accumulation_steps=2" "TrainingArguments.learning_rate=0.00001" "TrainingArguments.num_train_epochs=5" "TrainingArguments.lr_scheduler_type='cosine'" "TrainingArguments.warmup_steps=10000"

#Finetune Libriaug in GSS
#ginpipe configs/base/train_asr.gin \
#        configs/datasets/chime6_mixer6_dipco_gss.gin \
#        configs/models/wavlm_rnnt.gin \
#        configs/models/training_config.gin \
#        --module_list configs/imports \
#        --project_name chime_asrs \
#        --experiment_name gss-wavlm-large-rnnt-libriaug-300k-finetunegss-lr1e5-warmupcosinedecay-6conformerlayers \
#        --mods "MODEL_PATH='/mnt/matylda5/qpepino/baseline-asr/experiments/chime_asrs/gss-wavlm-large-rnnt-libriaug/checkpoint-300000'" "TrainingArguments.eval_steps=1000" "TrainingArguments.per_device_train_batch_size=8" "TrainingArguments.gradient_accumulation_steps=2" "TrainingArguments.learning_rate=0.00001" "TrainingArguments.num_train_epochs=5" "TrainingArguments.lr_scheduler_type='cosine'" "TrainingArguments.warmup_steps=10000" ""

#Finetune Libriaug in GSS
#ginpipe configs/base/train_asr.gin \
#        configs/datasets/chime6_mixer6_dipco_gss.gin \
#        configs/models/wavlm_rnnt.gin \
#        configs/models/training_config.gin \
#        --module_list configs/imports \
#        --project_name chime_asrs \
#        --experiment_name gss-wavlm-large-rnnt-libriaug-200k-finetunegss-lr1e4 \
#        --mods "MODEL_PATH='/mnt/matylda5/qpepino/baseline-asr/experiments/chime_asrs/gss-wavlm-large-rnnt-libriaug/checkpoint-300000'" "TrainingArguments.eval_steps=10000" "TrainingArguments.learning_rate=0.0001" "TrainingArguments.per_device_train_batch_size=8" "TrainingArguments.gradient_accumulation_steps=1"

#Multichannel Training
#Finetune Librimix in Multichannel
ginpipe configs/base/train_asr_torch.gin \
        configs/datasets/chime6_mixer6_dipco_multichannel.gin \
        configs/models/wavlm_rnnt_xv_multich.gin \
        configs/models/training_config.gin \
        --module_list configs/imports \
        --project_name chime_asrs \
        --experiment_name gss-wavlm-large-rnnt-multichannel-lr1e5-xch22-24-lr1e4 \
        --mods "MODEL_PATH='/mnt/matylda5/qpepino/baseline-asr/experiments/chime_asrs/gss-wavlm-large-rnnt-librimix/checkpoint-300000'" "TrainingArguments.eval_steps=10000" "WavLMRNNT.speaker_embedding_mode='sum'" "TrainingArguments.per_device_train_batch_size=2" "TrainingArguments.gradient_accumulation_steps=8" "TrainingArguments.learning_rate=0.0001" "TrainingArguments.num_train_epochs=100"

#Eval ASR torch dataset
#ginpipe configs/base/eval_asr_torch.gin \
#        configs/datasets/chime6_mixer6_dipco_torch.gin \
#        configs/models/wavlm_rnnt.gin \
#        --module_list configs/imports \
#        --project_name chime_asrs \
#        --experiment_name eval-asr-bestmodel-beamsize16 \
#        --mods "MODEL_PATH='/mnt/matylda5/qpepino/baseline-asr/experiments/chime_asrs/gss-wavlm-large-rnnt-encnotfrozen/checkpoint-130000'" "tasks.eval_model.split=['mixer6_dev','dipco_dev','chime6_dev']" "tasks.eval_model.beam_size=16"