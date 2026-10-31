#include "tissue_sound_detector.h"
#include "tissue_model_data.h"
#include "tissue_mel_filterbank.h"

#include <cmath>
#include <cstring>
#include <algorithm>

#include <esp_log.h>
#include <esp_heap_caps.h>
#include <freertos/FreeRTOS.h>
#include <freertos/task.h>
#include <freertos/semphr.h>

#include "model_meta.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

#define TAG "TissueDetect"

TissueSoundDetector& GetTissueSoundDetector() {
    static TissueSoundDetector instance;
    return instance;
}

static constexpr int kWin = TISSUE_WIN;          // 512
static constexpr int kHop = TISSUE_HOP;          // 160（10ms）
static constexpr int kFrames = TISSUE_FRAMES;    // 64
static constexpr int kMels = TISSUE_MEL_BINS;    // 40
static constexpr int kFftBins = TISSUE_FFT_BINS; // 257
static constexpr float kInScale = MODEL_IN_SCALE;
static constexpr int   kInZp = MODEL_IN_ZP;
static constexpr float kOutScale = MODEL_OUT_SCALE;
static constexpr int   kOutZp = MODEL_OUT_ZP;
static constexpr float kLogEps = 1e-6f;

// 基-2 复数 FFT（N=512）。精度已对照 numpy.fft 校验；
// 如需进一步省 CPU 可换 esp-dsp 的 dsps_fft2r_fc32（见 INTEGRATION.md）。
static void Fft512(float* re, float* im) {
    for (int i = 1, j = 0; i < 512; ++i) {
        int bit = 256;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) { std::swap(re[i], re[j]); std::swap(im[i], im[j]); }
    }
    for (int len = 2; len <= 512; len <<= 1) {
        int half = len >> 1;
        for (int i = 0; i < 512; i += len) {
            for (int k = 0; k < half; ++k) {
                float ang = -2.0f * static_cast<float>(M_PI) * k / len;
                float wr = cosf(ang), wi = sinf(ang);
                float ur = re[i + k], ui = im[i + k];
                float xr = re[i + k + half], xi = im[i + k + half];
                float vr = xr * wr - xi * wi;
                float vi = xr * wi + xi * wr;
                re[i + k] = ur + vr;        im[i + k] = ui + vi;
                re[i + k + half] = ur - vr; im[i + k + half] = ui - vi;
            }
        }
    }
}

bool TissueSoundDetector::Initialize() {
    if (interpreter_ != nullptr) return true;

    auto* model = tflite::GetModel(g_tissue_model_data);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        ESP_LOGE(TAG, "Model schema version mismatch");
        return false;
    }

    // 模型实际算子（已用 _get_ops_details() 核实）：
    // CONV_2D x3、MUL x3、ADD x3（BN 量化折叠）、FULLY_CONNECTED x2、
    // MAX_POOL_2D、MEAN（GAP）、LOGISTIC
    auto* resolver = new tflite::MicroMutableOpResolver<7>();
    resolver->AddConv2D();
    resolver->AddMul();
    resolver->AddAdd();
    resolver->AddFullyConnected();
    resolver->AddMaxPool2D();
    resolver->AddMean();
    resolver->AddLogistic();
    resolver_ = resolver;

    constexpr size_t kArenaSize = 64 * 1024;
    tensor_arena_ = static_cast<uint8_t*>(heap_caps_malloc(kArenaSize, MALLOC_CAP_SPIRAM));
    if (tensor_arena_ == nullptr)
        tensor_arena_ = static_cast<uint8_t*>(heap_caps_malloc(kArenaSize, MALLOC_CAP_INTERNAL));
    auto* interpreter = new tflite::MicroInterpreter(model, *resolver, tensor_arena_, kArenaSize);
    if (interpreter->AllocateTensors() != kTfLiteOk) {
        ESP_LOGE(TAG, "AllocateTensors failed（arena 不足时调大 kArenaSize）");
        delete interpreter;
        return false;
    }
    interpreter_ = interpreter;

    for (int i = 0; i < kWin; ++i)
        hann_[i] = 0.5f - 0.5f * cosf(2.0f * static_cast<float>(M_PI) * i / (kWin - 1));
    mel_.reserve(kFrames * kMels);
    buf_.reserve(kWindowSamples + 4096);
    fft_re_ = static_cast<float*>(heap_caps_malloc(kWin * sizeof(float), MALLOC_CAP_INTERNAL));
    fft_im_ = static_cast<float*>(heap_caps_malloc(kWin * sizeof(float), MALLOC_CAP_INTERNAL));
    mutex_ = xSemaphoreCreateMutex();

    if (xTaskCreate([](void* arg) {
            static_cast<TissueSoundDetector*>(arg)->AnalyzeLoop();
            vTaskDelete(nullptr);
        }, "tissue_det", 4096, this, 1, nullptr) != pdPASS) {
        ESP_LOGE(TAG, "Failed to create detector task");
        return false;
    }
    ESP_LOGI(TAG, "Tissue detector ready（model %u bytes, thr %.2f, %d/%d 窗）",
             g_tissue_model_data_len, kScoreThreshold, kHitsRequired, kHistorySize);
    return true;
}

void TissueSoundDetector::Enable(bool on) {
    bool was = enabled_.exchange(on);
    if (was == on) return;
    if (!on) {
        // 退出待机：丢弃残留音频，清空判决历史（避免上次对话的残响触发误唤醒）
        if (xSemaphoreTake(static_cast<SemaphoreHandle_t>(mutex_), pdMS_TO_TICKS(200)) == pdTRUE) {
            buf_.clear();
            xSemaphoreGive(static_cast<SemaphoreHandle_t>(mutex_));
        }
        hist_len_ = 0;
        hist_head_ = 0;
        refractory_ = 0;
        last_score_ = 0.0f;
    }
}

void TissueSoundDetector::Feed(const int16_t* samples, size_t count) {
    if (!enabled_.load() || interpreter_ == nullptr || samples == nullptr || count == 0) return;
    if (xSemaphoreTake(static_cast<SemaphoreHandle_t>(mutex_), 0) != pdTRUE) return;
    for (size_t i = 0; i < count; ++i) buf_.push_back(samples[i] / 32768.0f);
    xSemaphoreGive(static_cast<SemaphoreHandle_t>(mutex_));
}

void TissueSoundDetector::AnalyzeLoop() {
    while (true) {
        vTaskDelay(pdMS_TO_TICKS(50));
        if (!enabled_.load()) continue;
        bool has_window = false;
        if (xSemaphoreTake(static_cast<SemaphoreHandle_t>(mutex_), pdMS_TO_TICKS(100)) == pdTRUE) {
            has_window = buf_.size() >= kWindowSamples;
            xSemaphoreGive(static_cast<SemaphoreHandle_t>(mutex_));
        }
        if (!has_window) continue;

        AnalyzeWindow();  // ~10ms（含 64 次 FFT），在低优先级任务里不挤占音频路径

        if (xSemaphoreTake(static_cast<SemaphoreHandle_t>(mutex_), pdMS_TO_TICKS(100)) == pdTRUE) {
            buf_.erase(buf_.begin(), buf_.begin() + kSlideSamples);
            xSemaphoreGive(static_cast<SemaphoreHandle_t>(mutex_));
        }
    }
}

void TissueSoundDetector::AnalyzeWindow() {
    mel_.clear();
    for (int k = 0; k < kFrames; ++k) {
        const float* x = buf_.data() + k * kHop;
        for (int i = 0; i < kWin; ++i) {
            fft_re_[i] = x[i] * hann_[i];
            fft_im_[i] = 0.0f;
        }
        Fft512(fft_re_, fft_im_);
        float mel_row[kMels];
        for (int m = 0; m < kMels; ++m) {
            const float* row = &g_tissue_mel_filterbank[m * kFftBins];
            float acc = 0.0f;
            for (int b = 0; b < kFftBins; ++b) {
                float p = fft_re_[b] * fft_re_[b] + fft_im_[b] * fft_im_[b];
                acc += row[b] * p;
            }
            mel_row[m] = log10f(acc + kLogEps);
        }
        for (int m = 0; m < kMels; ++m) mel_.push_back(mel_row[m]);
    }

    // 整窗 mean/var 归一化（与训练一致）
    float mean = 0.0f, var = 0.0f;
    for (float v : mel_) mean += v;
    mean /= mel_.size();
    for (float v : mel_) var += (v - mean) * (v - mean);
    var /= mel_.size();
    const float inv_std = 1.0f / (sqrtf(var) + 1e-6f);

    auto* interpreter = static_cast<tflite::MicroInterpreter*>(interpreter_);
    TfLiteTensor* input = interpreter->input(0);
    TfLiteTensor* output = interpreter->output(0);
    for (size_t i = 0; i < mel_.size(); ++i) {
        float xq = std::clamp(roundf((mel_[i] - mean) * inv_std / kInScale) + kInZp, -128.0f, 127.0f);
        input->data.int8[i] = static_cast<int8_t>(xq);
    }
    if (interpreter->Invoke() != kTfLiteOk) {
        ESP_LOGW(TAG, "Invoke failed");
        return;
    }
    last_score_ = (output->data.int8[0] - kOutZp) * kOutScale;
    PushScore(last_score_);
}

void TissueSoundDetector::PushScore(float s) {
    if (refractory_ > 0) { --refractory_; return; }  // 冷却按窗计数（每窗 0.5s），与轮询频率无关
    history_[hist_head_] = s;
    hist_head_ = (hist_head_ + 1) % kHistorySize;
    if (hist_len_ < kHistorySize) ++hist_len_;
}

bool TissueSoundDetector::MaybeTrigger() {
    if (!enabled_.load() || !ready()) return false;
    if (hist_len_ < kHistorySize) return false;
    int hits = 0;
    for (int i = 0; i < kHistorySize; ++i)
        if (history_[i] >= kScoreThreshold) ++hits;
    if (hits >= kHitsRequired) {
        refractory_ = kRefractoryWindows;
        ESP_LOGI(TAG, "Tissue pull detected（%d/%d 窗命中）→ WakeWordInvoke", hits, kHistorySize);
        return true;
    }
    return false;
}
