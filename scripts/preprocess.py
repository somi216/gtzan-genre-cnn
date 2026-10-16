import sys, glob, os, yaml
import numpy as np
import librosa
from tqdm import tqdm

cfg = yaml.safe_load(open(sys.argv[1]))["data"]
os.makedirs(cfg["cache_dir"], exist_ok = True)

files = sorted(glob.glob(f"{cfg['raw_dir']}/*/*.wav"))
for f in tqdm(files):
    out = os.path.join(cfg["cache_dir"],
                       os.path.basename(f).replace(".wav", ".npy"))
    if os.path.exists(out):
        continue
    try:
        y, _ = librosa.load(f,  sr = cfg["sr"], duration = 29.5)
    except Exception as e:
        print("skip",f,e)
        continue
    mel = librosa.feature.melspectrogram(
        y=y, sr = cfg["sr"], n_fft = cfg["n_fft"],
        hop_length = cfg["hop"], n_mels = cfg["n_mels"])
    np.save(out,np.log(mel + 1e-6).astype(np.float32))