#!/usr/bin/env python3
"""生成训练数据：
- 负样本：真实环境音效（EigenScape CC-BY 合成类别，泼水/沙沙/纸张类相近声），滑窗切段 + 音量随机。
- 正样本：暂用 3~5 段"纸面摩擦 + 撕裂"素材长条（你拿到手后替换成真抽纸录音）滑窗切段 + 增广。
输出：data/train/{positive,negative}/*.wav (16kHz mono)
"""
import numpy as np, scipy.io.wavfile as wav, os, requests, io, random
import soundfile as sf

os.makedirs('data/train/positive', exist_ok=True)
os.makedirs('data/train/negative', exist_ok=True)

# EigenScape（CC-BY 4.0, ARU - University of Salford）合成声音类别样本
CLIPS = {
    'WaatMJKTw': 'pour',      # 泼水（水声类近似）
    'WaatMJKTj': 'spray',     # 喷洒
    'WaatMJKNh': 'garden',    # 沙沙（草木摩擦）
    'WaatMJKMz': 'bang',      # 碰撞
    'WaatMJKMs': 'crunch',    # 嘎吱/碎裂
}
BASE = 'https://zenodo.org/records/14863398/files/{clip}--{take}.wav?download=1'
rng = random.Random(42)

def load_long(url, take):
    r = requests.get(url, timeout=120)
    r.raise_for_status()
    x, sr = sf.read(io.BytesIO(r.content), dtype='float32', always_2d=True)
    return x[:, 0], sr

def rand_gain(x, lo=0.3, hi=1.4):
    return x * np.float32(rng.uniform(lo, hi))

def slices(x, sr, dur=1.0, n=None):
    L = int(sr * dur); out = []
    if len(x) < L: return out
    step = L // 2
    idx = list(range(0, len(x) - L, step)); rng.shuffle(idx)
    for i in (idx if n is None else idx[:n]):
        seg = x[i:i+L].copy()
        # 头尾 10ms 淡入淡出，避免边界爆音
        f = int(0.01 * sr)
        seg[:f] *= np.linspace(0, 1, f, dtype='float32')
        seg[-f:] *= np.linspace(1, 0, f, dtype='float32')
        out.append(seg)
    return out

def save(seg, sr, path):
    wav.write(path, sr, (np.clip(seg, -1, 1) * 32767).astype(np.int16))

def extract_events(x, sr, dur=1.0):
    """按能量事件切分：稀疏抽纸录音（抽一下、隔几秒）专用。
    返回 (positive_slices, local_negative_slices)
    - 正样本：含真实抽纸声的窗口（能量超过本录音噪声底 3 倍）
    - 本地负样本：完全安静的长间隙 —— 与部署环境同噪声底，比 ESC-50 更贴合
    """
    frame = int(0.1 * sr)
    n = len(x) // frame
    if n < 12:
        return [], []
    e = np.array([float(np.sqrt(np.mean(x[i*frame:(i+1)*frame]**2))) for i in range(n)])
    floor = float(np.percentile(e, 20))
    thr = max(floor * 3.0, 3e-4)
    active = (e > thr).astype(np.int8)
    # 膨胀 active 区域 ±0.5s，避免把抽纸声边缘切进负样本
    m = 5
    act = active.copy()
    for i in np.where(active)[0]:
        act[max(0, i-m):min(n, i+m+1)] = 1
    L = int(sr * dur); hop = L // 2
    # 强事件参考能量：用活跃帧的 75 分位代表“真抽纸”力度，
    # 过滤开头的指尖轻碰等弱事件（避免弱噪声被标成正样本）
    strong_ref = float(np.percentile(e[active], 75)) if active.any() else 0.0
    slice_accept = strong_ref * 0.25 if strong_ref > 0 else thr
    pos, neg = [], []
    min_neg_frames = 12  # 连续 1.2s 以上无事件才算可靠安静段
    i = 0
    while i + L <= len(x):
        seg = x[i:i+L]
        f0, f1 = i // frame, (i + L) // frame
        seg_peak = float(e[f0:f1].max()) if f1 > f0 else 0.0
        seg_active = act[f0:f1].any()
        if seg_active and seg_peak >= slice_accept:
            pos.append(seg)
        elif not seg_active and (f1 - f0) >= min_neg_frames:
            neg.append(seg)
        i += hop
    return pos, neg

n_pos = 0
# ---- 正样本 ----（真实稀疏录音：事件级切分）
import glob as _glob
raw_files = sorted(_glob.glob('data/raw/positive/*.wav')) + sorted(_glob.glob('data/raw/positive/*.mp3'))
if not raw_files:
    print('[skip] data/raw/positive/ 不存在或为空（等你的真实抽纸录音）')
for path in raw_files:
    x, sr = sf.read(path, dtype='float32', always_2d=True)
    x = x[:, 0]
    if sr != 16000:
        import scipy.signal as ss
        x = ss.resample_poly(x, 16000, sr); sr = 16000
    pos_segs, neg_segs = extract_events(x, sr)
    tag = os.path.basename(path).rsplit('.', 1)[0][:12]
    for i, seg in enumerate(pos_segs):
        save(rand_gain(seg), sr, f'data/train/positive/pos_{tag}_{i:03d}.wav')
        n_pos += 1
    for i, seg in enumerate(neg_segs):
        save(seg, sr, f'data/train/negative/neg_local_{tag}_{i:03d}.wav')
    print(f'{path}: 正样本 {len(pos_segs)} 段 / 本地负样本 {len(neg_segs)} 段')
print('positive clips:', n_pos)

# ---- 负样本 ----
n_neg = 0
for clip, tag in CLIPS.items():
    for take in ('DPA_M', 'MS_RB'):
        url = BASE.format(clip=clip, take=take)
        try:
            x, sr = load_long(url, take)
        except Exception as e:
            print(f'[warn] {url}: {e}'); continue
        if sr != 16000:
            import scipy.signal as ss
            x = ss.resample_poly(x, 16000, sr); sr = 16000
        segs = slices(x, sr, n=25)
        for i, seg in enumerate(segs):
            save(rand_gain(seg, 0.25, 1.2), sr,
                 f'data/train/negative/neg_{tag}_{take.split("_")[0]}_{i:03d}.wav')
            n_neg += 1
print('negative clips:', n_neg)
