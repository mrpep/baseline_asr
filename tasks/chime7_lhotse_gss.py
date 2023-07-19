import os
os.environ['HF_DATASETS_OFFLINE '] = "1"

import gzip
import json
import soundfile as sf
import numpy as np
import datasets 
from typing import Iterable, Tuple, Union, List, Dict
from argparse import ArgumentParser, Namespace
from pathlib import Path
from IPython import embed

def load_gzjson(filename):
    results = []
    with gzip.open(filename,'r') as f:
        for x in f.read().splitlines():
            results.append(json.loads(x))
    return results

class Chime7LhotseGSS(datasets.GeneratorBasedBuilder):
    def __init__(self, metadata_dir: os.PathLike, sampling_rate: int = 16000, blacklist=None, train_split='chime6-dev', eval_split='chime6-eval', **kwargs):
        super().__init__(**kwargs)
        self.sampling_rate = sampling_rate
        self.data_dir = metadata_dir
        self.blacklist = blacklist
        self.train_split = train_split
        self.eval_split = eval_split

    def _info(self):
        return datasets.DatasetInfo(
            description="",
            supervised_keys=None,
            homepage="",
        )  # TODO: Add dataset description and metadata

    def _prepare_split_single(
            self,
            gen_kwargs: dict,
            fpath: str,
            file_format: str,
            max_shard_size: int,
            split_info: datasets.SplitInfo,
            check_duplicate_keys: bool,
            job_id: int,
    ) -> Iterable[Tuple[int, bool, Union[int, tuple]]]:
        self.info.features = None  # Disable unnecessary type check and conversion that slows generation
        return super()._prepare_split_single(gen_kwargs, fpath, file_format, max_shard_size, split_info,
                                             check_duplicate_keys,
                                             job_id)

    def _fetch_split_meta(self, split: list):
        
        recordings_metadata = {}
        supervisions_metadata = {}
        
        if split == 'train':            
            split = self.train_split            
        elif split=='eval':
            split=self.eval_split
        if not isinstance(split, list):
            split = [split]
        
        for s in split:
            metadata_path = Path(self.data_dir,s.split('_')[0],s.split('_')[1])
            
            recordings = load_gzjson(Path(metadata_path,'{}_gss_recordings.jsonl.gz'.format(s.replace('-','_'))))
            recordings_metadata.update({x['id']: x for x in recordings})
            split_metadata = {s:load_gzjson(Path(metadata_path,'{}_gss_supervisions.jsonl.gz'.format(s.replace('-','_'))))}

            supervisions_metadata.update(split_metadata)

        return {'recordings_metadata': recordings_metadata, 'supervisions_metadata': supervisions_metadata}

    def _split_generators(self, _):
        """Generate dataset splits"""
        ds = [datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs=self._fetch_split_meta("train"))]
        for s in self.eval_split:
            ds += [datasets.SplitGenerator(name=s, gen_kwargs=self._fetch_split_meta(s))]
        return ds

    def _generate_examples(self, recordings_metadata, supervisions_metadata):
            for key in supervisions_metadata.keys():
                try:
                    for si in supervisions_metadata[key]:
                        recording_id = si['recording_id']
                        si_out = []
                        for c in si['channel']:
                            fs = recordings_metadata[recording_id]['sampling_rate']
                            for s in recordings_metadata[recording_id]['sources']:
                                if s['channels'] == [c]:
                                    p = Path('/mnt/matylda3/karafiat/GIT/CHiME-7/ASR/espnet.v0',s['source'])
                                    wav_i, fs = sf.read(p,start=int(si['start']*fs),stop=int((si['start']+si['duration'])*fs))
                                    si_out.append(wav_i)
                        si_out = np.stack(si_out)
                        #store only one channel si_out[0]
                        out_i = {'audio': {'array': si_out[0], 'sampling_rate': fs}, 'file': s['source'], 'text': si['text']}
                        if si_out.shape[1]//320 > 1.5*len(si['text']):
                            yield si['id'], out_i
                        else:
                            print('Discarding sample {} as it is not long enough compared to the transcription'.format(si['recording_id']))
                except:
                    print('Failed reading {}'.format(si['recording_id']))
                    continue

def parse_args() -> Namespace:
    parser = ArgumentParser(
        description='Preprocessing script to generate Chime7 dataset.')
    parser.add_argument('dataset', type=str, help='')
    parser.add_argument('metadata_dir', type=str, help='')
    parser.add_argument('dataset_cache', type=str, help='')
    parser.add_argument('train_split', type=str, help='', default='chime6-dev')
    parser.add_argument('eval_split', type=str, help='', default='chime6-eval')
    parser.add_argument('--num_proc', type=int, default=1, help='')
    parser.add_argument('--blacklist', type=str, help='')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    if not isinstance(args.train_split, list):
        args.train_split=[args.train_split]

    dataset = datasets.load_dataset(args.dataset, keep_in_memory=False, cache_dir=args.dataset_cache,
                                   metadata_dir=args.metadata_dir, num_proc=args.num_proc, train_split=args.train_split, eval_split=args.eval_split)

    #dataset = datasets.load_dataset(args.dataset, keep_in_memory=False, cache_dir=args.dataset_cache,metadata_dir=args.metadata_dir, num_proc=args.num_proc)

    #if not os.path.exists('datasets'):
    #    os.mkdir('datasets')
    
    #embed()

    #dataset.save_to_disk('datasets/')
