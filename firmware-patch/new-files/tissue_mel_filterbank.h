#pragma once
#include <cstdint>
#define TISSUE_MEL_BINS 40
#define TISSUE_FFT_BINS 257
#define TISSUE_WIN 512
#define TISSUE_HOP 160
#define TISSUE_FRAMES 64
#define TISSUE_SR 16000
extern const float g_tissue_mel_filterbank[40 * 257];
