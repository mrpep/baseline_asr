ginpipe configs/base/train_asr.gin \
        configs/datasets/timit.gin \
        configs/models/wav2vec2.gin \
        configs/models/training_config.gin \
        --module_list configs/imports \
        --project_name timit_asrs \
        --experiment_name w2v2_base

ginpipe configs/base/train_asr.gin \
        configs/datasets/timit.gin \
        configs/models/wav2vec2.gin \
        configs/models/training_config.gin \
        --module_list configs/imports \
        --project_name timit_asrs \
        --experiment_name w2v2_base-960h \
        --mods "MODEL_PATH='/mnt/matylda5/qpepino/hf_models/wav2vec2-base-960h'"

ginpipe configs/base/train_asr.gin \
        configs/datasets/timit.gin \
        configs/models/wavlm.gin \
        configs/models/training_config.gin \
        --module_list configs/imports \
        --project_name timit_asrs \
        --experiment_name wavlm-base

ginpipe configs/base/train_asr.gin \
        configs/datasets/chime.gin \
        configs/models/wav2vec2.gin \
        configs/models/training_config.gin \
        --module_list configs/imports \
        --project_name chime_asrs \
        --experiment_name w2v2_base-960h \
        --mods "MODEL_PATH='/mnt/matylda5/qpepino/hf_models/wav2vec2-base-960h'"

ginpipe configs/base/train_asr.gin \
        configs/datasets/chime6_mixer6_dipco_gss.gin \
        configs/models/wav2vec2.gin \
        configs/models/training_config.gin \
        --module_list configs/imports \
        --project_name chime_asrs \
        --experiment_name gss-w2v2_base-960h \
        --mods "MODEL_PATH='/mnt/matylda5/qpepino/hf_models/wav2vec2-base-960h'"

ginpipe configs/base/train_asr.gin \
        configs/datasets/chime6_mixer6_dipco_gss.gin \
        configs/models/wavlm.gin \
        configs/models/training_config.gin \
        --module_list configs/imports \
        --project_name chime_asrs \
        --experiment_name gss-wavlm-large \
        --mods "MODEL_PATH='/mnt/matylda5/qpepino/hf_models/wavlm-large'" "TrainingArguments.per_device_train_batch_size=8" "TrainingArguments.gradient_accumulation_steps=4"

ginpipe configs/base/train_asr.gin \
        configs/datasets/chime6_mixer6_dipco_gss.gin \
        configs/models/wav2vec2.gin \
        configs/models/training_config.gin \
        --module_list configs/imports \
        --project_name chime_asrs \
        --experiment_name gss-wavlm-large-960h-lv60-self \
        --mods "MODEL_PATH='/mnt/matylda5/qpepino/hf_models/wav2vec2-large-960h-lv60-self'" "TrainingArguments.per_device_train_batch_size=8" "TrainingArguments.gradient_accumulation_steps=4"

ginpipe configs/base/train_asr.gin \
        configs/datasets/chime6_mixer6_dipco_gss.gin \
        configs/models/wavlm.gin \
        configs/models/training_config.gin \
        --module_list configs/imports \
        --project_name chime_asrs \
        --experiment_name gss-wavlm-large-notfrozenenc \
        --mods "MODEL_PATH='/mnt/matylda5/qpepino/hf_models/wavlm-large'" "TrainingArguments.per_device_train_batch_size=8" "TrainingArguments.gradient_accumulation_steps=4" "tasks.load_hf_model.freeze_feature_extractor=False"

ginpipe configs/base/train_asr.gin \
        configs/datasets/chime6_mixer6_dipco_gss.gin \
        configs/models/wavlm_rnnt.gin \
        configs/models/training_config.gin \
        --module_list configs/imports \
        --project_name chime_asrs \
        --experiment_name gss-wavlm-base-rnnt \
        --mods "MODEL_PATH='/mnt/matylda5/qpepino/hf_models/wavlm-base'"

ginpipe configs/base/train_asr.gin \
        configs/datasets/chime6_mixer6_dipco_gss.gin \
        configs/models/wavlm_rnnt.gin \
        configs/models/training_config.gin \
        --module_list configs/imports \
        --project_name chime_asrs \
        --experiment_name gss-wavlm-large-rnnt \
        --mods "MODEL_PATH='/mnt/matylda5/qpepino/hf_models/wavlm-large'" "TrainingArguments.eval_steps=10000"

#EVAL

ginpipe configs/base/eval_asr.gin \
        configs/datasets/chime6_mixer6_dipco_gss.gin \
        configs/models/wavlm_rnnt.gin \
        configs/models/training_config.gin \
        --module_list configs/imports \
        --project_name chime_asrs \
        --experiment_name gss-wavlm-large-rnnt-eval \
        --mods "MODEL_PATH='/mnt/matylda5/qpepino/baseline-asr/experiments/chime_asrs/gss-wavlm-large-rnnt/checkpoint-10500'"