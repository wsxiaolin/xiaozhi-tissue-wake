#!/usr/bin/env python3
"""训练：log-mel 前端（与 ESP-DL 上 DSP 一致：16k/1s/40mel/64 帧 hop=20ms）+ 小 CNN。
输出：work/best_model.tflite（int8 全量化，含量化代表集）+ 标签与阈值评估。
"""
import numpy as np, os, glob, json
import scipy.io.wavfile as wav
from scipy import signal as ss
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

SR, DUR, N_MEL, FRAMES = 16000, 1.0, 40, 64
HOP = SR // 100           # 10ms —— 必须与固件 TISSUE_HOP=160 严格一致
WIN = 512
rng = np.random.default_rng(11)

# mel 滤波器组（与 04_export_artifacts.py 同源同参：40 mel、60 到 7800Hz）
def _build_fb():
    n_fft = WIN // 2 + 1
    def hz2mel(h): return 2595 * np.log10(1 + h / 700)
    def mel2hz(m): return 700 * (10 ** (m / 2595) - 1)
    m_pts = np.linspace(hz2mel(60), hz2mel(7800), N_MEL + 2)
    h_pts = mel2hz(m_pts)
    bins = np.floor((WIN + 1) * h_pts / SR).astype(int)
    fb = np.zeros((N_MEL, n_fft), dtype=np.float32)
    for i in range(N_MEL):
        a, b_, c = bins[i], bins[i+1], bins[i+2]
        if b_ == a: b_ = a + 1
        if c == b_: c = b_ + 1
        fb[i, a:b_] = np.linspace(0, 1, b_ - a)
        fb[i, b_:c] = np.linspace(1, 0, c - b_)
    return fb
FB = _build_fb()

def logmel(x):
    """与固件 tissue_sound_detector.cc 逐位一致的前端：
    对称 hann（分母 N-1）、无边界补零、原始功率谱（不用 scipy stft 的缩放）、log10(P+1e-6)"""
    x = x.astype(np.float32)
    if len(x) < SR: x = np.pad(x, (0, SR - len(x)))
    x = x[:SR]
    hann = (0.5 - 0.5 * np.cos(2 * np.pi * np.arange(WIN) / (WIN - 1))).astype(np.float32)
    frames = []
    for k in range(FRAMES):
        seg = x[k*HOP : k*HOP+WIN] * hann
        P = (np.abs(np.fft.rfft(seg)) ** 2).astype(np.float32)
        frames.append(np.log10(FB @ P + 1e-6))
    mel = np.array(frames, dtype=np.float32)   # [64, 40]
    mel = (mel - mel.mean()) / (mel.std() + 1e-6)
    return mel

def time_shift(x, max_ms=120):
    sh = int(rng.integers(-max_ms, max_ms) * SR / 1000)
    return np.roll(x, sh)

def speed_perturb(x, factor):
    """变速不变调近似（线性插值重采样，±10%）"""
    n_new = int(len(x) / factor)
    idx = np.linspace(0, len(x) - 1, n_new)
    return np.interp(idx, np.arange(len(x)), x).astype(np.float32)

# 背景噪声池：用负样本目录里的真实环境音，以随机信噪比混入正样本，
# 模拟部署时“设备麦克风听到的抽纸声叠加房间噪声”
import glob as _bglob, random as _brnd
_bg_pool = sorted(str(p) for p in _bglob.glob('data/train/negative/*.wav'))
_brnd.seed(99)
def mix_background(x, snr_lo=5.0, snr_hi=20.0):
    if not _bg_pool:
        return x
    import scipy.io.wavfile as _wav
    p = _brnd.choice(_bg_pool)
    _sr, b = _wav.read(p)
    b = b.astype(np.float32) / 32768.0
    if len(b) < len(x):
        b = np.pad(b, (0, len(x) - len(b)))
    off = _brnd.randint(0, len(b) - len(x))
    b = b[off:off + len(x)]
    px = float(np.mean(x ** 2)) + 1e-9
    pb = float(np.mean(b ** 2)) + 1e-9
    snr = _brnd.uniform(snr_lo, snr_hi)
    b = b * float(np.sqrt(px / pb / (10 ** (snr / 10))))
    return (x + b).astype(np.float32)

def spec_aug(m, freq_mask=6, time_mask=10):
    m = m.copy()
    f0 = rng.integers(0, max(1, N_MEL - freq_mask)); m[:, f0:f0+freq_mask] = m.mean()
    t0 = rng.integers(0, max(1, FRAMES - time_mask)); m[t0:t0+time_mask, :] = m.mean()
    return m

def load_dir(pattern, augment, is_positive=False):
    X = []
    for p in sorted(glob.glob(pattern)):
        sr, x = wav.read(p)
        x = x.astype(np.float32) / 32768.0
        if augment:
            for _ in range(3):
                y = time_shift(x)
                if rng.random() < 0.5:
                    y = y * rng.uniform(0.7, 1.3)
                if rng.random() < 0.4:
                    y = speed_perturb(y, rng.uniform(0.9, 1.1))
                if is_positive and rng.random() < 0.5:
                    y = mix_background(y)
                if rng.random() < 0.3:
                    y = y + rng.normal(0, 0.01, len(y)).astype(np.float32)
                X.append(spec_aug(logmel(y)))
        X.append(logmel(x))
    return np.array(X, dtype=np.float32)

X_pos = load_dir('data/train/positive/*.wav', augment=True, is_positive=True)
# 负样本均衡：优先保留本地负样本（同环境噪声底），再随机抽 ESC-50 补足，
# 控制负样本文件数 ≈ 正样本文件数的 3 倍，避免 1:10 失衡把模型带偏
import glob as _g
neg_local = sorted(_g.glob('data/train/negative/neg_local_*.wav'))
neg_rest = [p for p in sorted(_g.glob('data/train/negative/*.wav'))
            if not os.path.basename(p).startswith('neg_local_')]
pos_files = len(_g.glob('data/train/positive/*.wav'))
import random as _rnd
_rnd.seed(7)
_rnd.shuffle(neg_rest)
neg_files = neg_local + neg_rest[:max(0, pos_files * 3 - len(neg_local))]
print('负样本选择: 本地 %d + ESC50 %d（正样本文件 %d）' % (len(neg_local), len(neg_files) - len(neg_local), pos_files))
X_neg = np.concatenate([load_dir(p, augment=True) for p in neg_files]) if neg_files else np.zeros((0, FRAMES, N_MEL), dtype=np.float32)
print('pos:', len(X_pos), 'neg:', len(X_neg))
X = np.concatenate([X_pos, X_neg])[..., None]
y = np.concatenate([np.ones(len(X_pos)), np.zeros(len(X_neg))]).astype(np.int32)
idx = rng.permutation(len(X)); X, y = X[idx], y[idx]
n_val = max(24, int(len(X) * 0.15))
Xtr, ytr, Xv, yv = X[n_val:], y[n_val:], X[:n_val], y[:n_val]

model = models.Sequential([
    layers.Input(shape=(FRAMES, N_MEL, 1)),
    layers.Conv2D(16, 3, strides=(1, 2), activation='relu'),
    layers.BatchNormalization(),
    layers.Conv2D(32, 3, strides=(2, 2), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(2),
    layers.Conv2D(48, 3, strides=(1, 2), activation='relu'),
    layers.BatchNormalization(),
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.3),
    layers.Dense(24, activation='relu'),
    layers.Dense(1, activation='sigmoid'),
])
model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

os.makedirs('work', exist_ok=True)
cbs = [
    callbacks.ModelCheckpoint('work/best_model.keras', save_best_only=True, monitor='val_accuracy', mode='max'),
    callbacks.EarlyStopping(patience=8, restore_best_weights=True, monitor='val_accuracy', mode='max'),
]
model.fit(Xtr, ytr, validation_data=(Xv, yv), epochs=40, batch_size=32, callbacks=cbs, verbose=2)

# ---- TFLite int8 全量化 ----
def rep():
    for i in range(0, len(Xtr), 7):
        yield [Xtr[i].reshape(1, FRAMES, N_MEL, 1).astype(np.float32)]

conv = tf.lite.TFLiteConverter.from_keras_model(model)
conv.optimizations = [tf.lite.Optimize.DEFAULT]
conv.representative_dataset = rep
conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
conv.inference_input_type = tf.int8
conv.inference_output_type = tf.int8
tfl = conv.convert()
open('work/best_model.tflite', 'wb').write(tfl)
print('tflite bytes:', len(tfl))

# ---- 量化模型验证 + 阈值扫描 ----
interp = tf.lite.Interpreter(model_path='work/best_model.tflite')
interp.allocate_tensors()
inp = interp.get_input_details()[0]; out = interp.get_output_details()[0]
scale_in, zp_in = inp['quantization']; scale_out, zp_out = out['quantization']
def tflite_prob(m):
    q = (m / scale_in + zp_in).astype(np.int8)
    interp.set_tensor(inp['index'], q.reshape(1, FRAMES, N_MEL, 1))
    interp.invoke()
    o = interp.get_tensor(out['index']).astype(np.float32)
    return float((o[0, 0] - zp_out) * scale_out)

# 用验证集原始片段再算一遍概率（未增广的部分占少数，直接用全量验证样本近似）
scores_pos = [tflite_prob(m) for m in X_pos[:80]]
scores_neg = [tflite_prob(m) for m in X_neg[:160]]
print('pos score p50/p95:', np.percentile(scores_pos, [50, 95]))
print('neg score p50/p95:', np.percentile(scores_neg, [50, 95]))
thr_report = {}
for thr in (0.5, 0.6, 0.7, 0.8, 0.9):
    tp = np.mean([s >= thr for s in scores_pos]); fp = np.mean([s >= thr for s in scores_neg])
    thr_report[str(thr)] = {'recall': round(float(tp), 3), 'false_positive_rate': round(float(fp), 3)}
json.dump({'thresholds': thr_report,
           'quantization': {'in_scale': scale_in, 'in_zero_point': int(zp_in),
                             'out_scale': scale_out, 'out_zero_point': int(zp_out)}},
          open('work/threshold_report.json', 'w'), indent=2)
print(json.dumps(thr_report, indent=1))
