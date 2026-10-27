import sys, yaml, random, os, time
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from dataset import split_files, MelSegmentDataset
from model import GenreCNN


def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main(cfg_path):
    cfg = yaml.safe_load(open(cfg_path))
    set_seed(cfg["seed"])
    d, t = cfg["data"], cfg["train"]
    seg_frames = int(d["seg_sec"] * d["sr"] / d["hop"])

    (trf, trl), (vaf, val), _, genres = split_files(d["cache_dir"], cfg["seed"])

    # train 데이터의 통계로만 정규화값 계산
    sample = np.concatenate([np.load(f)[:, ::10].ravel() for f in trf])
    mean, std = float(sample.mean()), float(sample.std())

    tr_ds = MelSegmentDataset(trf, trl, seg_frames, True, mean, std,
                          repeat=t.get("repeat", 1))
    va_ds = MelSegmentDataset(vaf, val, seg_frames, False, mean, std)
    tr_ld = DataLoader(tr_ds, t["batch_size"], shuffle=True,
                       num_workers=t["num_workers"])
    va_ld = DataLoader(va_ds, t["batch_size"],
                       num_workers=t["num_workers"])

    device = get_device()
    print("device:", device)
    model = GenreCNN(len(genres)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=t["lr"])
    loss_fn = nn.CrossEntropyLoss()

    run = time.strftime("%Y%m%d-%H%M%S")
    out = f"outputs/{run}"
    os.makedirs(out)
    writer = SummaryWriter(out)
    yaml.dump(cfg, open(f"{out}/config.yaml", "w"))

    best = 0
    for ep in range(t["epochs"]):
        model.train()
        tl = 0
        for xb, yb in tr_ld:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            tl += loss.item()

        model.eval()
        correct = n = 0
        with torch.no_grad():
            for xb, yb in va_ld:
                xb, yb = xb.to(device), yb.to(device)
                correct += (model(xb).argmax(1) == yb).sum().item()
                n += len(yb)
        acc = correct / n

        writer.add_scalar("loss/train", tl / len(tr_ld), ep)
        writer.add_scalar("acc/val", acc, ep)
        print(f"ep {ep+1} | loss {tl/len(tr_ld):.3f} | val {acc:.3f}")

        if acc > best:
            best = acc
            torch.save({"model": model.state_dict(), "mean": mean,
                        "std": std, "genres": genres}, f"{out}/best.pt")


if __name__ == "__main__":
    main(sys.argv[1])