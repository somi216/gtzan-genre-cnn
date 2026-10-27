import sys, os, yaml
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from dataset import split_files
from model import GenreCNN


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def predict_song(model, mel, seg, hop, mean, std, device):
    """곡 하나를 여러 구간으로 잘라 softmax 확률을 평균"""
    T = mel.shape[1]
    starts = range(0, T - seg + 1, hop)
    segs = np.stack([mel[:, s:s + seg] for s in starts])
    x = torch.from_numpy((segs - mean) / std).float().unsqueeze(1).to(device)
    with torch.no_grad():
        probs = model(x).softmax(1).mean(0)
    return probs.cpu().numpy()


def evaluate(model, files, labels, seg, hop, mean, std, device):
    preds = []
    for f in files:
        mel = np.load(f)
        preds.append(predict_song(model, mel, seg, hop, mean, std, device).argmax())
    preds = np.array(preds)
    labels = np.array(labels)
    return (preds == labels).mean(), labels, preds


def main(run_dir):
    cfg = yaml.safe_load(open(f"{run_dir}/config.yaml"))
    d = cfg["data"]
    seg = int(d["seg_sec"] * d["sr"] / d["hop"])
    hop = seg // 2                      # 50% 겹치게 자름

    ckpt = torch.load(f"{run_dir}/best.pt", map_location="cpu")
    mean, std, genres = ckpt["mean"], ckpt["std"], ckpt["genres"]

    device = get_device()
    model = GenreCNN(len(genres))
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()

    _, (vaf, val), (tef, tel), _ = split_files(d["cache_dir"], cfg["seed"])

    va_acc, _, _ = evaluate(model, vaf, val, seg, hop, mean, std, device)
    te_acc, labels, preds = evaluate(model, tef, tel, seg, hop, mean, std, device)

    print(f"run: {run_dir}")
    print(f"val  song acc: {va_acc:.3f}  (모델 선택에 쓴 데이터라 낙관적)")
    print(f"test song acc: {te_acc:.3f}  <- 최종 baseline")

    print("\n장르별 test 정확도")
    for i, g in enumerate(genres):
        m = labels == i
        if m.sum():
            print(f"  {g:10s} {(preds[m] == i).mean():.2f}  ({m.sum()}곡)")

    cm = confusion_matrix(labels, preds)
    fig, ax = plt.subplots(figsize=(8, 8))
    ConfusionMatrixDisplay(cm, display_labels=genres).plot(
        ax=ax, xticks_rotation=45, colorbar=False)
    plt.tight_layout()
    out = f"{run_dir}/confusion_test.png"
    plt.savefig(out, dpi=150)
    print(f"\n혼동 행렬 저장: {out}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        run = sys.argv[1]
    else:
        runs = sorted(os.listdir("outputs"))
        run = f"outputs/{runs[-1]}"     # 가장 최근 실험
    main(run)
