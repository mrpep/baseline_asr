import glob
import joblib

all_chime = glob.glob('experiments/chime_asrs/gss-wavlm-large-rnnt-deveval-29500-beam20/mixer6_deveval-*.pkl')
all_d = [joblib.load(f) for f in all_chime]

chime_dict = {}
for d in all_d:
    for k,v in d.items():
        if k!='wer':
            if k in chime_dict:
                chime_dict[k].extend(v)
            else:
                chime_dict[k] = v