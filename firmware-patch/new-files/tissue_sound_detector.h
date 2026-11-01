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
// 判决（v5，用户反馈“太钝”，全面放宽）：
//   滑动间隔 0.25s，最近 16 窗（4 秒）：
//   ① 强规则：任一窗 score ≥ 0.70（一次明确抽纸，比 v4 的 0.78 更灵敏）→ 触发
//   ② 弱规则：≥3 窗 score ≥ 0.50（两次不太明确的抽纸累积）→ 触发
//   触发后冷却 24 窗（6 秒）。
class TissueSoundDetector {
public:
    static constexpr float  kStrongScore     = 0.70f;  // ① 一次明确抽纸的门槛（v5: 0.78→0.70）
    static constexpr float  kMediumScore     = 0.50f;  // ② 模糊抽纸的计分门槛（v5: 0.55→0.50）
    static constexpr int    kMediumHits      = 3;      // ② 16 窗内命中次数（v5: 4→3，对应两下）
    static constexpr int    kHistorySize     = 16;     // 16 × 0.25s = 4 秒跨度
    static constexpr int    kRefractoryWindows = 24;   // 每窗 0.25s → 冷却 6s（v5: 10s→6s）
    static constexpr size_t kWindowSamples   = 16000;  // 1s 分析窗
    static constexpr size_t kSlideSamples    = 4000;   // 0.25s 滑动

    bool Initialize();   // 加载模型、建分析任务；成功后 ready()==true
    // 任意任务线程安全；从 AFE 旁路喂 16k 单声道 PCM（任意块大小）
    void Feed(const int16_t* samples, size_t count);
    // 周期调用（1s 一次即可）；满足判决条件返回 true（内部自带冷却）
    bool MaybeTrigger();
    void Enable(bool on);   // 只在待机态监听：非 idle 由 Application 关闭，避免自己 TTS 触发自己
    bool IsActive() const { return enabled_.load(); }
    float last_score() const { return last_score_; }
    bool ready() const { return interpreter_ != nullptr; }

private:
    void AnalyzeLoop();          // 分析任务主体
    void AnalyzeWindow(std::vector<float>& win);  // 分析一个 16000 样本窗口快照
    void PushScore(float s);

    std::vector<float> buf_;            // 原始 PCM 池（mutex_ 保护）
    std::vector<float> work_;           // 窗口快照（仅分析任务使用，锁外分析用）
    std::vector<float> mel_;            // 64*40 展平
    float hann_[512];
    float history_[kHistorySize] = {0};
    int   hist_len_ = 0, hist_head_ = 0;
    int   refractory_ = 0;
    float last_score_ = 0.0f;

    void* task_handle_ = nullptr;
    void* mutex_ = nullptr;             // SemaphoreHandle_t
    std::atomic<bool> enabled_{false};
    std::atomic<bool> in_analysis_{false};  // Enable(false) 等待分析轮次结束的代际标志
    int zero_window_count_ = 0;             // 聱启动自检：连续全零窗计数
    float* fft_re_ = nullptr;           // 工作区（内部 RAM 一次性分配）
    float* fft_im_ = nullptr;

    // TFLite Micro
    uint8_t* tensor_arena_ = nullptr;
    void* resolver_ = nullptr;
    void* interpreter_ = nullptr;
};

// 全局实例访问器（Application 与 AfeAudioEngine 旁路共用）
TissueSoundDetector& GetTissueSoundDetector();
