#!/usr/bin/bash
set -x

__conda_setup="$('/mnt/matylda5/qpepino/miniconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)" ; eval "$__conda_setup"
. "/mnt/matylda5/qpepino/miniconda3/etc/profile.d/conda.sh"
unset __conda_setup
unset PYTHONHOME
conda activate multichannel_wavlm

cd /mnt/matylda5/qpepino/baseline-asr

ginpipe configs/base/eval_asr.gin \
        configs/datasets/chime6_mixer6_dipco_gss_deveval.gin \
        configs/models/wavlm_rnnt.gin \
        configs/models/training_config.gin \
        --module_list configs/imports \
        --project_name chime_asrs \
        --experiment_name gss-wavlm-large-rnnt-deveval-29500-beam20 \
        --mods "MODEL_PATH='/mnt/matylda5/qpepino/baseline-asr/experiments/chime_asrs/gss-wavlm-large-rnnt/ckpt29500'" "tasks.eval_model.beam_size=20" "tasks.eval_model.start=$1" "tasks.eval_model.end=$2"
