\
import argparse, os, glob, yaml
import numpy as np

def load_yaml(p):
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def load_labels(label_dir):
    wh = []
    for p in glob.glob(os.path.join(label_dir, "*.txt")):
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                s = line.strip().split()
                if len(s) >= 5:
                    wh.append([float(s[3]), float(s[4])])
    return np.array(wh, dtype=np.float32)

def kmeans(wh, n=9, iters=1000, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(wh), size=n, replace=False)
    c = wh[idx].copy()
    for _ in range(iters):
        w = wh[:,0:1]; h = wh[:,1:2]
        cw = c[:,0][None,:]; ch = c[:,1][None,:]
        inter = np.minimum(w,cw)*np.minimum(h,ch)
        union = (w*h)+(cw*ch)-inter+1e-9
        iou = inter/union
        d = 1-iou
        a = d.argmin(axis=1)
        new = np.stack([wh[a==i].mean(axis=0) if np.any(a==i) else c[i] for i in range(n)], axis=0)
        if np.allclose(new,c,atol=1e-4): break
        c = new
    return c

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--img", type=int, default=1024)
    ap.add_argument("--n", type=int, default=9)
    args = ap.parse_args()
    d = load_yaml(args.data)
    root = d["path"]
    label_dir = d["train"].replace("images","labels")
    label_dir = label_dir if os.path.isabs(label_dir) else os.path.join(root, label_dir)
    wh = load_labels(label_dir)
    if len(wh)==0:
        raise SystemExit("No labels found.")
    wh = wh * args.img
    a = kmeans(wh, n=args.n)
    a = a[np.argsort(a.prod(axis=1))]
    print("Anchors (px):")
    for x in a:
        print(f"  - [{x[0]:.1f}, {x[1]:.1f}]")
