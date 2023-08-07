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
        --experiment_name gss-wavlm-large-rnnt-debug \
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

#Finetune encoder

ginpipe configs/base/train_asr.gin \
        configs/datasets/chime6_mixer6_dipco_gss.gin \
        configs/models/wavlm_rnnt.gin \
        configs/models/training_config.gin \
        --module_list configs/imports \
        --project_name chime_asrs \
        --experiment_name gss-wavlm-large-rnnt-encnotfrozen-lr1e4 \
        --mods "MODEL_PATH='/mnt/matylda5/qpepino/hf_models/wavlm-large'" "TrainingArguments.eval_steps=10000" "tasks.load_hf_model.freeze_feature_extractor=False" "TrainingArguments.learning_rate=0.0001" "TrainingArguments.per_device_train_batch_size=4" "TrainingArguments.gradient_accumulation_steps=4"

####XVector conditioning####

#This is concatenation + linear proj
ginpipe configs/base/train_asr.gin \
        configs/datasets/chime6_mixer6_dipco_gss_xv.gin \
        configs/models/wavlm_rnnt_xv.gin \
        configs/models/training_config.gin \
        --module_list configs/imports \
        --project_name chime_asrs \
        --experiment_name gss-wavlm-large-rnnt-xv \
        --mods "MODEL_PATH='/mnt/matylda5/qpepino/hf_models/wavlm-large'" "TrainingArguments.eval_steps=10000" "TrainingArguments.per_device_train_batch_size=8" "TrainingArguments.gradient_accumulation_steps=2"

#This is projection of xvector + sum with weight (initially 0)
ginpipe configs/base/train_asr.gin \
        configs/datasets/chime6_mixer6_dipco_gss_xv.gin \
        configs/models/wavlm_rnnt_xv.gin \
        configs/models/training_config.gin \
        --module_list configs/imports \
        --project_name chime_asrs \
        --experiment_name gss-wavlm-large-rnnt-xv-sum \
        --mods "MODEL_PATH='/mnt/matylda5/qpepino/hf_models/wavlm-large'" "TrainingArguments.eval_steps=10000" "WavLMRNNT.speaker_embedding_mode='sum'"

#Finetune encoder lr=1e-5 + augmented data
ginpipe configs/base/train_asr.gin \
        configs/datasets/librispeech_500_chime7.gin \
        configs/models/wavlm_rnnt.gin \
        configs/models/training_config.gin \
        --module_list configs/imports \
        --project_name chime_asrs \
        --experiment_name gss-wavlm-large-rnnt-encnotfrozen-lr1e5-augmenteddata \
        --mods "MODEL_PATH='/mnt/matylda5/qpepino/hf_models/wavlm-large'" "TrainingArguments.eval_steps=10000" "tasks.load_hf_model.freeze_feature_extractor=False" "TrainingArguments.learning_rate=0.00001" "TrainingArguments.per_device_train_batch_size=4" "TrainingArguments.gradient_accumulation_steps=4"
