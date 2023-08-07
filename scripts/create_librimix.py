import soundfile as sf
from IPython.display import Audio
from pathlib import Path
from tqdm import tqdm
import pandas as pd
from functools import partial
import random
import numpy as np
import librosa
import pedalboard
import fire
import joblib
import multiprocessing as mp

def add_noise(audio, noise_files, prob=0.8):
    dice = random.uniform(0,1)
    if dice < prob:
        gain = random.uniform(0,0.4)
        idx = random.randint(0,len(noise_files)-1)
        noise, fs = librosa.core.load(noise_files[idx], sr=16000)
        if noise.shape[0]>audio.shape[0]:
            start_idx = random.randint(0,noise.shape[0]-audio.shape[0])
            noise = noise[start_idx:start_idx+audio.shape[0]]
        else:
            noise = np.tile(noise,(audio.shape[0]//noise.shape[0] + 1,))
            noise = noise[:audio.shape[0]]

        return (1-gain)*audio + gain*noise
    else:
        return audio
    
def reverberate(audio, rir_files, prob=0.8):
    dice = random.uniform(0,1)
    if dice < prob:
        idx = random.randint(0,len(rir_files)-1)
        convolver = pedalboard.Convolution(str(rir_files[idx].resolve()))
        return convolver(audio, 16000)
    else:
        return audio
    
def generate_mixture(audios, single_audio_processing_fns=None, mix_processing_fns=None):
    mix = None
    alphas = np.random.uniform(0.5,1,size=len(audios))
    alphas = alphas/np.sum(alphas)
    for i,f in enumerate(audios):
        x, fs = sf.read(f)
        if single_audio_processing_fns is not None:
            for fn in single_audio_processing_fns:
                x = fn(x)
                
        if mix is None:
            mix = x*alphas[0]
        else:
            if x.shape[0] < mix.shape[0]:
                start_idx = random.randint(0,mix.shape[0]-x.shape[0])
                mix[start_idx:start_idx+x.shape[0]] += x*alphas[i]
            else:
                start_idx = random.randint(0,x.shape[0]-mix.shape[0])
                x = x*alphas[i]
                x[start_idx:start_idx+mix.shape[0]] += mix
                mix = x
    if mix_processing_fns is not None:
        for fn in mix_processing_fns:
            mix = fn(mix)
            
    return mix

def mix_speakers(df,N=1000,max_speakers=1, output_path=None, 
                 single_audio_processing_fns=None, 
                 mix_processing_fns=None,
                 offset=0,
                 supervision_filename='supervisions.pkl'):

    id_by_spk = {spk: df.loc[df['speaker_id']==spk] for spk in df['speaker_id'].unique()}
    speakers = np.array(list(id_by_spk.keys()))
    mixs = []
    supervisions = []
    if not Path(output_path).exists():
        Path(output_path).mkdir(parents=True)
    for i in tqdm(range(N)):
        i+=offset
        try:
            n_spk = random.randint(1,max_speakers)
            spks = np.random.choice(speakers, n_spk, replace=False)
            rows = [id_by_spk[spk].sample(1) for spk in spks]
            mix_filename = Path(output_path,'{}_mix_'.format(i) + '_'.join([r.index[0] for r in rows]) + '.wav')
            supervision_i = [{'transcription': r['transcription'].values[0], 
                             'speaker_id': r['speaker_id'].values[0],
                             'filename': mix_filename} for r in rows]
            audios = [r['filename'].values[0] for r in rows]
            mix = generate_mixture(audios, single_audio_processing_fns=single_audio_processing_fns, mix_processing_fns=mix_processing_fns)
            supervisions.extend(supervision_i)
            sf.write(str(mix_filename.resolve()), mix, 16000)
        except:
            print('Failed {}'.format(i))
    joblib.dump(supervisions,Path(output_path,supervision_filename))
    return supervisions

def mix_speakers_i(offset,s_filename, librispeech_df=None, rir_fn=None, add_noise_fn=None, out_dir=None, N=0):
    return mix_speakers(librispeech_df, N=N, single_audio_processing_fns=[rir_fn],
                        mix_processing_fns=[add_noise_fn],output_path=out_dir,
                        supervision_filename=s_filename,
                        offset=offset)
def create_librimix(N,
                    out_dir,
                    librispeech_path='/mnt/matylda2/data/LibriSpeech/LibriSpeech/train-clean-360', 
                    rir_dir='/mnt/matylda2/data/RIRS_NOISES/simulated_rirs/smallroom', 
                    noise_dir='/mnt/matylda2/data/NOISES_MUSIC/WHAM', 
                    offset=0,
                    supervision_offset=0,
                    n_proc=4):
    rows = []
    for x in tqdm(Path(librispeech_path).rglob('*.flac')):
        metadata = {'chapter': x.parts[-2],
                    'speaker_id': x.parts[-3],
                    'filename': str(x.resolve()),
                    'id': x.stem}
        rows.append(metadata)
        
    trans = []
    for x in tqdm(Path(librispeech_path).rglob('*.trans.txt')):
        with open(x,'r') as f:
            all_lines = f.readlines()
            for l in all_lines:
                l.replace('\n','')
                l_parts = l.split()
                trans.append({'idx': l_parts[0],
                            'transcription': ' '.join(l_parts[1:])})
                
    trans_df = pd.DataFrame(trans)
    audio_df = pd.DataFrame(rows)
    librispeech_df =  pd.merge(trans_df, audio_df, left_on='idx', right_on='id')
    librispeech_df = librispeech_df.set_index('idx')

    add_noise_fn = partial(add_noise, noise_files=list(Path(noise_dir).rglob('*.wav')))
    rir_fn = partial(reverberate, rir_files=list(Path(rir_dir).rglob('*.wav')))

    mix_speakers_fn = partial(mix_speakers_i, 
                              librispeech_df=librispeech_df, 
                              N=N//n_proc, 
                              rir_fn=rir_fn,
                              add_noise_fn=add_noise_fn,
                              out_dir=out_dir)
    
    offsets = [offset+i for i in range(0,N,N//n_proc)]
    supervision_filenames = ['supervision_{}.pkl'.format(i+supervision_offset) for i in range(len(offsets))]
    
    with mp.Pool(n_proc) as pool:
        pool.starmap(mix_speakers_fn,list(zip(offsets,supervision_filenames)))

if __name__ == '__main__':
    fire.Fire(create_librimix)