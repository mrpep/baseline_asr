import torch
from pathlib import Path
import joblib
from tqdm import tqdm
import soundfile as sf

class Chime7Dataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path, processor):
        super().__init__()
        supervision_list = Path(dataset_path).rglob('*.pkl')
        all_supervisions = []
        for s in tqdm(supervision_list, desc='Loading metadata'):
            all_supervisions.extend(joblib.load(s))
        self.all_supervisions = all_supervisions
        self.processor = processor

    def __getitem__(self, idx):
        si = self.all_supervisions[idx]
        x, fs = sf.read(si['filename'])

        return {'wav': x}

    def __len__(self):
        return len(self.all_supervisions)
