# INTEGRATION.md · 把抽纸检测接入 xiaozhi-esp32（commit f443114 基线）

> 目标：抽纸声 → 触发唤醒 → 走官方链路（语音唤醒词、ASR/LLM/TTS、MCP 全部不受影响）。
> 前提：ESP32-S3 + PSRAM 板型（同多唤醒词要求），ESP-IDF ≥ 6.0.1。

## 第 0 步：拷文件

```
<xiaozhi-esp32>/main/audio/sound_events/
├── tissue_model_data.cc/.h
├── tissue_mel_filterbank.cc/.h
├── model_meta.h
└── tissue_sound_detector.h/.cc
```

## 第 1 步：加入构建（main/CMakeLists.txt）

在 `SOURCES` 列表追加：
```cmake
    "audio/sound_events/tissue_sound_detector.cc"
    "audio/sound_events/tissue_model_data.cc"
    "audio/sound_events/tissue_mel_filterbank.cc"
```
并确保 `PRIV_REQUIRES` 有 `esp-dsp`（可选：内置 radix-2 FFT 已可直接用，esp-dsp 仅作提速；`idf_component.yml` 加 `espressif/esp-dsp: ^1.5.0`）。**固件 FFT 已有可用实现**（`tissue_sound_detector.cc` 内 `Fft512`，精度已对照 numpy 校验）；如需省 CPU 再换 esp-dsp：
```cpp
#include <dsps_fft2r_fc32.h>
// 初始化一次：dsps_fft2r_init_fc32(NULL, 0);
// 帧处理：dsps_fft2r_fc32(re, 512); dsps_bit_rev_fc32(re, 512);
// power[i] = re[2i]^2 + re[2i+1]^2  （esp-dsp 输出实部虚部交错）
```
另：`espressif/esp-tflite-micro` 也必须加入依赖（提供 TFLM 运行时），`firmware/model_meta.h` 由导出脚本自动生成与模型同步。

FFT 512 点 float 在 240MHz S3 上约 15 到 20 微秒/帧（esp-dsp），整窗 64 帧 + CNN 约 6 到 10ms，每 0.5s 一次，占用 <2% 单核（内置 FFT 约 2 倍耗时，同样可用）。

## 第 2 步：Kconfig 开关（main/Kconfig.projbuild）

```
menu "Sound Event Wake"
    config SOUND_EVENT_TISSUE
        bool "Enable tissue-pull sound wake"
        default n
    config SOUND_EVENT_TISSUE_THRESHOLD
        int "Score threshold (%)"
        range 50 99
        default 90
    config SOUND_EVENT_TISSUE_STREAK
        int "Consecutive windows required"
        range 1 8
        default 3
endmenu
```
（默认值读取进 detector 初始化，替代硬编码，方便现场调参不用重训。）

## 第 3 步：挂进 Application（main/application.cc）

初始化（`Initialize()` 末尾）：
```cpp
#if CONFIG_SOUND_EVENT_TISSUE
    tissue_detector_ = std::make_unique<TissueSoundDetector>();
    tissue_detector_->Initialize();
    // 500ms 周期喂音：直接复用音频服务现有通路最稳——见第 4 步
#endif
```

触发动作（任意线程安全，因为 `WakeWordInvoke` 内部会 Schedule）：
```cpp
#if CONFIG_SOUND_EVENT_TISSUE
    if (tissue_detector_ && tissue_detector_->MaybeTrigger()) {
        WakeWordInvoke("抽纸");   // 后续与语音唤醒完全同路：Connecting→Listening→官方对话
    }
#endif
```

## 第 4 步：拿到音频数据（两种方案二选一）

**方案 A（推荐，改动最小）**：idle 状态本来就开着唤醒检测，AFE 在跑。
在 `AfeAudioEngine::HandleWakeWordResult()` 的调用上游（`ProcessingTask()` 拿到
`result` 处）加一行旁路：把 `result->data`（16k mono int16）同时喂给 detector——
**不影响原有 WakeNet/MultiNet 判定**，纯旁路只读：
```cpp
// afe_audio_engine.cc ProcessingTask() 中 fetch 之后：
#if CONFIG_SOUND_EVENT_TISSUE
    extern TissueSoundDetector* GetTissueDetector();
    if (auto* d = GetTissueDetector()) {
        d->Feed(reinterpret_cast<const int16_t*>(result->data),
                result->data_size / sizeof(int16_t));
    }
#endif
```
注意 AFE 输出帧长是 feed chunksize（256/10ms），detector 内部按 160 hop 攒窗，
`Feed()` 已支持任意大小分块。

**方案 B（lite 引擎板型）**：在 `LiteAudioEngine` 的数据出口旁路同上。

## 第 5 步：定时判决

detector 每 0.5s 完整出一个窗分数。最简单：在 `Application::Run()` 的
`MAIN_EVENT_CLOCK_TICK`（1s 周期）里判两次，或在 audio_service 100ms 周期回调里判。
触发即调上面第 3 步的 `WakeWordInvoke`。

## 第 6 步：状态细节（重要）

- **Speaking/Listening 期间不触发**：外面包一层状态判断
  `if (GetDeviceState() == kDeviceStateIdle)` 才调 `WakeWordInvoke`——
  （`WakeWordInvoke` 本身对非 idle 状态有自己的处理，但抽纸场景限定 idle 更符合直觉。）
- **触发后的体验**：与喊"你好小智"完全一致（进入 Connecting → 上行 → 官方对话）。
  提示语里设备会念唤醒词；想自定义可传 `"抽纸"` 显示名。
- **误触兜底**：除模型+连窗双闸外，可在触发后加一次轻提示音（`PlaySound(OGG_POPUP)`）
  ——现有 `BeginWakeWordInvoke` 流程已含 popup 音，无需额外处理。

## 第 7 步：内存预算（S3-N16R8）

| 项 | 大小 |
|---|---|
| tflite 模型 | 30KB（flash，mmap 或拷 PSRAM） |
| tensor arena | 64KB（PSRAM） |
| 攒窗缓冲 16000×4B | 64KB（PSRAM） |
| mel 帧缓冲 64×40×4B | 10KB |
| FFT 工作区 | 2KB（内部 RAM） |
| **合计新增** | **≈170KB，其中内部 RAM < 8KB** |

与 AFE/WakeNet/LVGL 共存无压力（8MB PSRAM 板实测余量 2MB+）。

## 常见坑

1. **前端不一致 = 精度崩**：固件 mel 滤波器组必须用 `tissue_mel_filterbank.cc`
   （与训练同一份），归一化必须是"整窗 mean/var"，log 是 `log10(x+1e-6)`。
2. **FFT 溢出**：esp-dsp float32 无溢出问题；若换定点实现需重做量化校准。
3. **输入量化**：`x_int8 = round(x / 0.089352) + (-71)`，饱和到 [-128,127]
   —— `model_meta.h` 已给常量，重训后自动更新。
4. **重训后**：`04_export_artifacts.py` 会同时更新模型数组、mel 表（不变则同）、
   `model_meta.h`、`threshold_report.json`，全量替换 main/audio/sound_events/ 下文件再编译。
5. **实测偏差**：训练增益范围 0.7–1.3 覆盖一般电平；若你的录音音量普遍很低，
   在 `01_gen_dataset.py` 的 `rand_gain` 里扩大下限重跑。
