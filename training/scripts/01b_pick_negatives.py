#!/usr/bin/env python3
"""从已下载的 ESC-50 抽取负样本（需要先解压到 ../../.openclaw/tmp/ESC-50-master）。"""
import scipy.io.wavfile as wav
from scipy import signal as ss
import numpy as np, pandas as pd, os, glob, shutil

OUT = os.path.join(os.path.dirname(__file__), '..', 'data', 'train', 'negative')
OUT = os.path.abspath(OUT)
os.makedirs(OUT, exist_ok=True)
for f in glob.glob(OUT + '/*.wav'):
    os.remove(f)

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '.openclaw', 'tmp', 'ESC-50-master'))
meta = pd.read_csv(BASE + '/meta/esc50.csv')

# 相近混淆类：摩擦/沙沙/高频爆裂/水声 —— 全量
confusing = ['crackling_fire', 'brushing_teeth', 'pouring_water', 'water_drops',
             'door_wood_creaks', 'drinking_sipping', 'footsteps', 'keyboard_typing',
             'can_opening', 'glass_breaking']
# 通用环境类 —— 每类抽 10 条
generic = ['rain', 'wind', 'insects', 'chirping_birds', 'clock_tick', 'coughing',
           'breathing', 'sneezing', 'snoring', 'laughing', 'toilet_flush',
           'vacuum_cleaner', 'washing_machine', 'chainsaw', 'engine', 'helicopter',
           'train', 'car_horn']

rng = np.random.default_rng(3)
n = 0
def emit(category, rows, cap):
    global n
    for _, r in rows.head(cap).iterrows():
        sr, x = wav.read(os.path.join(BASE, 'audio', r.filename))
        x = x.astype(np.float32) / 32768.0
        if sr != 16000:
            x = ss.resample_poly(x, 16000, sr); sr = 16000
        a = int(1.5 * 16000)
        mid = x[a:a + int(2 * 16000)]
        for k, seg in enumerate((mid[:16000], mid[16000:])):
            if len(seg) < 16000:
                continue
            seg = seg * np.float32(rng.uniform(0.4, 1.1))
            wav.write(f'{OUT}/neg_{category}_{int(r.fold)}_{int(r.target)}_{k}.wav', sr,
                      (np.clip(seg, -1, 1) * 32767).astype(np.int16))
            n += 1

for c in confusing:
    emit(c, meta[meta.category == c], 40)
for c in generic:
    emit(c, meta[meta.category == c], 10)
print('negative clips written:', n)
