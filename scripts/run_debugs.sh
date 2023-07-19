#!/usr/bin/bash
set -x

#ginpipe configs/base/train_asr.gin \
#        configs/datasets/timit.gin \
#        configs/models/wav2vec2.gin \
#        configs/models/training_config.gin \
#        --module_list configs/imports \
#        --project_name timit_asrs \
#        --experiment_name timit_debug1

#WavLM Timit
#ginpipe configs/base/train_asr.gin \
#        configs/datasets/timit.gin \
#        configs/models/wavlm.gin \
#        configs/models/training_config.gin \
#        --module_list configs/imports \
#        --project_name timit_asrs \
#        --experiment_name wavlm_timit

ginpipe configs/base/train_asr.gin \
        configs/datasets/timit.gin \
        configs/models/wavlm.gin \
        configs/models/training_config.gin \
        --module_list configs/imports \
        --project_name timit_asrs \
        --experiment_name wavlm_timit_debug3 \
        --mods --mods "tasks.load_hf_processor.from_pretrained='patrickvonplaten/wavlm-libri-clean-100h-base-plus'"