#!/usr/bin/bash
set -x

#$ -N librimix
#$ -q long.q@@blade
#$ -l ram_free=32G,mem_free=32G,cpu=16
#$ -l matylda5=1,tmp_free=40G
#$ -o /mnt/matylda5/qpepino/librimix.o
#$ -e /mnt/matylda5/qpepino/librimix.e

# Initialize environment
__conda_setup="$('/mnt/matylda5/qpepino/miniconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)" ; eval "$__conda_setup"
. "/mnt/matylda5/qpepino/miniconda3/etc/profile.d/conda.sh"
unset __conda_setup
unset PYTHONHOME
conda activate multichannel_wavlm

cd /mnt/matylda5/qpepino/baseline-asr/scripts

python create_librimix.py 10000 /mnt/matylda5/qpepino/librispeech_augment --n_proc 16 --offset 300000 --supervision_offset 480