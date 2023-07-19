import datasets
from IPython.display import Audio
import numpy as np
import gzip
import json
import librosa
import numpy as np
import torch
from speechbrain.pretrained import EncoderClassifier
import joblib
from pathlib import Path
from tqdm import tqdm

def show_example(x):
    audio = np.array(x['audio']['array'])
    fs = x['audio']['sampling_rate']
    display(Audio(audio,rate=fs,normalize=False))
    print(x['text'])

def load_gzjson(filename):
    results = []
    with gzip.open(filename,'r') as f:
        for x in f.read().splitlines():
            results.append(json.loads(x))
    return results

def get_best_channels(x, recs, xvector_model, spk_xvectors):
    chs = get_all_channels(x, recs)
    xvs = [extract_xvector(xi, xvector_model) for xi in chs]
    target_xv = spk_xvectors[x['speaker']]
    scores = [np.dot(xv,target_xv)/(np.linalg.norm(xv)*np.linalg.norm(target_xv)) for xv in xvs]
    return scores.index(max(scores))
    
def get_all_channels(x, recs):
    s = x['start']
    d = x['duration']
    
    chs = [librosa.core.load(xi['source'], offset=s, duration=d, sr=None)[0] for xi in recs[x['recording_id']]['sources']]
    ch_lens = [chi.shape[0] for chi in chs]
    if len(set(ch_lens)) > 1:
        chs = [chi[:min(ch_lens)] for chi in chs]
    return np.stack(chs)

def extract_xvector(wav, model, device='cpu'):
    embeddings = model.encode_batch(torch.from_numpy(wav).to(device))
    return embeddings[0,0].detach().cpu().numpy()

def make_spk_to_xv_dict(sup_ihm, rec_ihm, model):
    spk_to_xv = {}
    for s in tqdm(sup_ihm):
        spk = s['speaker']
        rec_data = rec_ihm[s['recording_id']]
        sources = [source['source'] for source in rec_data['sources']]
        target_src = [src for src in sources if Path(src).stem.endswith(spk)]
        if len(target_src)>0:
            target_src = target_src[0]
        else:
            target_src = sources[0]
        x,fs = librosa.core.load(target_src,sr=None,offset=s['start'],duration=s['duration'])
        xv = extract_xvector(x, model)
        if spk in spk_to_xv:
            spk_to_xv[spk].append(xv)
        else:
            spk_to_xv[spk] = [xv]
    spk_to_xv = {k: np.mean(np.stack(v),axis=0) for k,v in spk_to_xv.items()}
    return spk_to_xv

model = EncoderClassifier.from_hparams(source="/mnt/matylda5/qpepino/pretrained_models/spkrec-xvect-voxceleb")
folders = ['mixer6']
for folder in folders:
    for i, split in enumerate(list(Path('/mnt/matylda3/karafiat/GIT/CHiME-7/ASR/espnet.v0/data/lhotse/{}'.format(folder)).glob('*'))):
        if split.parts[-1] != 'train':
            continue
        ihm_rec = list(Path(split).glob('*ihm_recordings*'))
        if len(ihm_rec)>0:
            ihm_rec = load_gzjson(ihm_rec[0])
            mdm_rec = load_gzjson(list(Path(split).glob('*mdm_recordings*'))[0])
            ihm_sup = load_gzjson(list(Path(split).glob('*ihm_supervisions*'))[0])
            mdm_sup = load_gzjson(list(Path(split).glob('*mdm_supervisions*'))[0])
            ihm_rec = {xi['id']: xi for xi in ihm_rec}
            mdm_rec = {xi['id']: xi for xi in mdm_rec}
            if Path('{}-{}-xv.pkl'.format(folder,split.parts[-1])).exists():
                spk_to_xv = joblib.load('{}-{}-xv.pkl'.format(folder,split.parts[-1]))
            else:
                spk_to_xv = make_spk_to_xv_dict(ihm_sup, ihm_rec, model)
                joblib.dump(spk_to_xv,'{}-{}-xv.pkl'.format(folder,split.parts[-1]))
            new_supervisions = []
            for s in tqdm(mdm_sup):
                best_ch = get_best_channels(s, mdm_rec, model, spk_to_xv)
                s['channel'] = [best_ch]
                new_supervisions.append(s)
            
            with gzip.open('{}-{}-supervised_bestchannel.jsonl.gz'.format(folder, split.parts[-1]), 'w') as f:
                f.write(json.dumps(new_supervisions).encode())
        else:
            print('No IHM recordings for {}'.format(split))
