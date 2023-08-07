#!/usr/bin/bash
set -x

#$ -N libriaug
#$ -q long.q@@gpu
#$ -l ram_free=64G,mem_free=64G,gpu=1,gpu_ram=46G,cpu=16
#$ -l matylda5=5,tmp_free=40G
#$ -o /mnt/matylda5/qpepino/baseline-asr/logs/gss-wavlm-large-rnnt-libriaug.o
#$ -e /mnt/matylda5/qpepino/baseline-asr/logs/gss-wavlm-large-rnnt-libriaug.e

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

ginpipe configs/base/train_asr_torch.gin \
        configs/datasets/librispeech_augmented.gin \
        configs/models/wavlm_rnnt.gin \
        configs/models/training_config.gin \
        --module_list configs/imports \
        --project_name chime_asrs \
        --experiment_name gss-wavlm-large-rnnt-libriaug \
        --mods "MODEL_PATH='/mnt/matylda5/qpepino/hf_models/wavlm-large'" "TrainingArguments.eval_steps=10000" "TrainingArguments.per_device_train_batch_size=8" "TrainingArguments.gradient_accumulation_steps=2" "tasks.load_hf_model.freeze_feature_extractor=False" "TrainingArguments.learning_rate=0.00001"

#ginpipe configs/base/train_asr_torch.gin \
#        configs/datasets/librimix.gin \
#        configs/models/wavlm_rnnt_xv.gin \
#        configs/models/training_config.gin \
#        --module_list configs/imports \
#        --project_name chime_asrs \
#        --experiment_name gss-wavlm-large-rnnt-librimix \
#        --mods "MODEL_PATH='/mnt/matylda5/qpepino/hf_models/wavlm-large'" "TrainingArguments.eval_steps=10000" "WavLMRNNT.speaker_embedding_mode='sum'" "TrainingArguments.per_device_train_batch_size=8" "TrainingArguments.gradient_accumulation_steps=2" "tasks.load_hf_model.freeze_feature_extractor=False" "TrainingArguments.learning_rate=0.00001"
