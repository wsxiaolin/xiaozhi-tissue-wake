#pragma once

#include <cstdint>
#include <cstddef>
#include <vector>
#include <atomic>
#include "model_meta.h"

// 抽纸声检测器（独立于 WakeNet/MultiNet 的旁路唤醒源）。
// 前端与训练脚本 03_train.py 逐位一致（已通过固件/训练双端校验）：
//   16k 单声道 → 64 帧 × hop160(10ms) × win512 对称 hann → rfft 功率 →
//   40 mel(60-7800Hz) → log10(P+1e-6) → 整窗 mean/var 归一化 → int8 TFLite。
//
// 判决：最近 8 窗（约 4 秒跨度）中 ≥5 窗 score ≥ 0.8 → 触发。
//   对应"连抽几张（3 秒上下连续抽纸声）才有反应"，单张短促抽纸不触发；
//   触发后冷却 10 秒。全部参数可在常量区调，无需重训。
class TissueSoundDetector {
public:
    static constexpr float  kScoreThreshold   = MODEL_SCORE_THRESH;  // 0.80
    static constexpr int    kHistorySize      = 8;
    static constexpr int    kHitsRequired     = 5;
    static constexpr int    kRefractoryWindows = 20;   // 每窗 0.5s → 冷却 10s
    static constexpr size_t kWindowSamples    = 16000; // 1s 分析窗
    static constexpr size_t kSlideSamples     = 8000;  // 0.5s 滑动

    bool Initialize();   // 加载模型、建分析任务；成功后 ready()==true
    // 任意任务线程安全；从 AFE 旁路喂 16k 单声道 PCM（任意块大小）
    void Feed(const int16_t* samples, size_t count);
    // 周期调用（1s 一次即可）；满足判决条件返回 true（内部自带冷却）
    bool MaybeTrigger();
    void Enable(bool on);   // 只在待机态监听：非 idle 由 Application 关闭，避免自己 TTS 触发自己
    float last_score() const { return last_score_; }
    bool ready() const { return interpreter_ != nullptr; }

private:
    void AnalyzeLoop();          // 分析任务主体
    void AnalyzeWindow();        // 消费 buf_ 前端 16000 样本 → 推理 → PushScore
    void PushScore(float s);

    std::vector<float> buf_;            // 原始 PCM 池（mutex_ 保护）
    std::vector<float> mel_;            // 64*40 展平
    float hann_[512];
    float history_[kHistorySize] = {0};
    int   hist_len_ = 0, hist_head_ = 0;
    int   refractory_ = 0;
    float last_score_ = 0.0f;

    void* task_handle_ = nullptr;
    void* mutex_ = nullptr;             // SemaphoreHandle_t
    std::atomic<bool> enabled_{false};
    float* fft_re_ = nullptr;           // 工作区（内部 RAM 一次性分配）
    float* fft_im_ = nullptr;

    // TFLite Micro
    uint8_t* tensor_arena_ = nullptr;
    void* resolver_ = nullptr;
    void* interpreter_ = nullptr;
};

// 全局实例访问器（Application 与 AfeAudioEngine 旁路共用）
TissueSoundDetector& GetTissueSoundDetector();
