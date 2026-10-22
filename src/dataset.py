import glob, os, random
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

def split_files(cache_dir, seed):
    files = sorted(glob.glob(f"{cache_dir}/*.npy"))
    genres = sorted({os.path.basename(f).split(".")[0] for f in files})
    labels = [genres.index(os.path.basename(f).split(".")[0]) for f in files]
    tr, tmp, ytr, ytmp = train_test_split(
        files, labels, test_size = 0.3, stratify = labels, random_state = seed)
    va, te, yva, yte = train_test_split(
        tmp, ytmp, test_size = 0.5, stratify = ytmp, random_state = seed)
    return (tr, ytr), (va, yva), (te, yte), genres

class MelSegmentDataset(Dataset):
    def __init__(self, files, labels, seg_frames, train, mean, std):
        self.files, self.labels = files, labels
        self.seg, self.train = seg_frames, train
        self.mean, self.std = mean, std
        self.mels = [np.load(f, mmap_mode="r") for f in files]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        mel = self.mels[i]
        T = mel.shape[1]
        if self.train:
            s = random.randint(0,T - self.seg)
        else:
            s = (T- self.seg)//2
        x = np.array(mel[:, s:s+self.seg])
        x = (x - self.mean)/self.std
        return torch.from_numpy(x)[None], self.labels[i]
    