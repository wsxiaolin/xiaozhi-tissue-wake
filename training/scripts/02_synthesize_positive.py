#!/usr/bin/env python3
"""正样本占位合成器：程序合成"抽纸"声（多段可变刚度摩擦 + 高频撕裂爆发 + 低频尾音）。
这是你拿到真实录音前的 pipeline 调试占位；正式训练请替换为 data/raw/positive/*.wav 真实录音。
"""
import numpy as np, scipy.io.wavfile as wav, os
from scipy import signal as ss

rng = np.random.default_rng(7)
SR = 16000
DUR = 1.0
N = 160

os.makedirs('data/train/positive', exist_ok=True)

def synth_tissue():
    L = int(SR * DUR)
    t = np.arange(L) / SR
    out = np.zeros(L, dtype=np.float32)
    # 3~6 个摩擦段：带通噪声包络（双手/纸盒摩擦的"沙沙"主体）
    for _ in range(rng.integers(3, 7)):
        start = rng.uniform(0, DUR - 0.25)
        dur = rng.uniform(0.08, 0.25)
        s0, s1 = int(start * SR), int((start + dur) * SR)
        n = s1 - s0
        noise = rng.normal(0, 1, n).astype(np.float32)
        b, a = ss.butter(2, [rng.uniform(900, 1600), rng.uniform(4000, 6500)], btype='band', fs=SR)
        seg = ss.lfilter(b, a, noise)
        env = np.exp(-np.linspace(0, rng.uniform(3, 8), n)).astype(np.float32)
        out[s0:s1] += seg * env * rng.uniform(0.2, 0.5)
    # 1~2 个"撕裂"爆发：短促高频 chirp+噪声簇
    for _ in range(rng.integers(1, 3)):
        start = rng.uniform(0.1, DUR - 0.2)
        dur = rng.uniform(0.05, 0.15)
        s0, s1 = int(start * SR), int((start + dur) * SR)
        n = s1 - s0
        f0 = rng.uniform(2500, 4000)
        chirp = ss.chirp(np.linspace(0, dur, n), f0, dur, f0 * rng.uniform(1.2, 2.0)).astype(np.float32)
        noise = rng.normal(0, 1, n).astype(np.float32) * 0.4
        env = (np.hanning(n) ** 0.5).astype(np.float32)
        out[s0:s1] += (chirp * 0.5 + noise) * env * rng.uniform(0.3, 0.6)
    # 低频盒体共鸣尾音
    n = int(0.12 * SR)
    s0 = int(rng.uniform(0.2, 0.6) * SR)
    body = np.sin(2 * np.pi * rng.uniform(120, 220) * np.arange(n) / SR).astype(np.float32)
    out[s0:s0+n] += body * np.exp(-np.linspace(0, 6, n)).astype(np.float32) * 0.15
    out *= rng.uniform(0.5, 1.0)
    f = int(0.01 * SR)
    out[:f] *= np.linspace(0, 1, f).astype(np.float32)
    out[-f:] *= np.linspace(1, 0, f).astype(np.float32)
    return out.astype(np.float32)

for i in range(N):
    x = synth_tissue()
    wav.write(f'data/train/positive/synth_{i:03d}.wav', SR,
              (np.clip(x, -1, 1) * 32767).astype(np.int16))
print('synthesized positive clips:', N)
