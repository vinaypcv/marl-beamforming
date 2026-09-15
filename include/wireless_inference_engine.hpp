#ifndef WIRELESS_INFERENCE_ENGINE_HPP
#define WIRELESS_INFERENCE_ENGINE_HPP

#include <cstdint>
#include <cmath>
#include <immintrin.h>

typedef int16_t q8_7_t;

#define FLOAT_TO_Q8_7(x) ((q8_7_t)((x) * 128.0f))
#define Q8_7_TO_FLOAT(x) ((float)(x) / 128.0f)

class EmbeddedUeActorEngine {
private:
    static const int TIME_STEPS = 8;
    static const int FEATURE_DIM = 16;
    static const int HIDDEN_DIM = 64;
    static const int OUTPUT_DIM = 16;

    alignas(32) q8_7_t temporal_processor_weight_ih_l0[3 * HIDDEN_DIM][FEATURE_DIM];
    alignas(32) q8_7_t temporal_processor_weight_hh_l0[3 * HIDDEN_DIM][HIDDEN_DIM];
    alignas(32) q8_7_t temporal_processor_bias_ih_l0[3 * HIDDEN_DIM];
    alignas(32) q8_7_t temporal_processor_bias_hh_l0[3 * HIDDEN_DIM];
    alignas(32) q8_7_t fc_policy_0_weight[32][HIDDEN_DIM];
    alignas(32) q8_7_t fc_policy_0_bias[32];
    alignas(32) q8_7_t fc_policy_2_weight[OUTPUT_DIM][32];
    alignas(32) q8_7_t fc_policy_2_bias[OUTPUT_DIM];

    q8_7_t tanh_lut[512];

    inline q8_7_t fast_tanh_lut(q8_7_t x) {
        int idx = (x + 256);
        if (idx < 0) idx = 0;
        if (idx > 511) idx = 511;
        return tanh_lut[idx];
    }

    inline q8_7_t fast_sigmoid_lut(q8_7_t x) {
        return (fast_tanh_lut(x >> 1) >> 1) + FLOAT_TO_Q8_7(0.5f);
    }

public:
    EmbeddedUeActorEngine() {
        for (int i = 0; i < 512; ++i) {
            float val = (i - 256) / 32.0f;
            tanh_lut[i] = FLOAT_TO_Q8_7(std::tanh(val));
        }

        for(int i=0; i < 3*HIDDEN_DIM; ++i) {
            for(int j=0; j < FEATURE_DIM; ++j) temporal_processor_weight_ih_l0[i][j] = FLOAT_TO_Q8_7(0.01f);
            for(int j=0; j < HIDDEN_DIM; ++j)  temporal_processor_weight_hh_l0[i][j] = FLOAT_TO_Q8_7(0.02f);
            temporal_processor_bias_ih_l0[i] = FLOAT_TO_Q8_7(0.001f);
            temporal_processor_bias_hh_l0[i] = FLOAT_TO_Q8_7(0.001f);
        }
        for(int i=0; i<32; ++i) {
            for(int j=0; j<HIDDEN_DIM; ++j) fc_policy_0_weight[i][j] = FLOAT_TO_Q8_7(0.05f);
            fc_policy_0_bias[i] = FLOAT_TO_Q8_7(0.01f);
        }
        for(int i=0; i<OUTPUT_DIM; ++i) {
            for(int j=0; j<32; ++j) fc_policy_2_weight[i][j] = FLOAT_TO_Q8_7(0.03f);
            fc_policy_2_bias[i] = FLOAT_TO_Q8_7(0.002f);
        }
    }

    int predict_next_beam(const q8_7_t* flattened_h_history) {
        q8_7_t h[HIDDEN_DIM] = {0};

        for (int t = 0; t < TIME_STEPS; ++t) {
            const q8_7_t* xt = flattened_h_history + (t * FEATURE_DIM);
            q8_7_t gate_inputs_ih[3 * HIDDEN_DIM] = {0};
            q8_7_t gate_inputs_hh[3 * HIDDEN_DIM] = {0};

            for (int i = 0; i < 3 * HIDDEN_DIM; ++i) {
                int32_t acc = 0;
                for (int j = 0; j < FEATURE_DIM; j += 16) {
                    __m256i v_w = _mm256_loadu_si256((const __m256i*)&temporal_processor_weight_ih_l0[i][j]);
                    __m256i v_x = _mm256_loadu_si256((const __m256i*)&xt[j]);
                    __m256i v_madd = _mm256_madd_epi16(v_w, v_x);
                    
                    alignas(32) int32_t temp[8];
                    _mm256_storeu_si256((__m256i*)temp, v_madd);
                    for(int k=0; k<8; ++k) acc += temp[k];
                }
                gate_inputs_ih[i] = (q8_7_t)(acc >> 7) + temporal_processor_bias_ih_l0[i];

                int32_t acc_h = 0;
                for (int j = 0; j < HIDDEN_DIM; j += 16) {
                    __m256i v_w = _mm256_loadu_si256((const __m256i*)&temporal_processor_weight_hh_l0[i][j]);
                    __m256i v_h = _mm256_loadu_si256((const __m256i*)&h[j]);
                    __m256i v_madd = _mm256_madd_epi16(v_w, v_h);

                    alignas(32) int32_t temp[8];
                    _mm256_storeu_si256((__m256i*)temp, v_madd);
                    for(int k=0; k<8; ++k) acc_h += temp[k];
                }
                gate_inputs_hh[i] = (q8_7_t)(acc_h >> 7) + temporal_processor_bias_hh_l0[i];
            }

            for (int d = 0; d < HIDDEN_DIM; ++d) {
                q8_7_t r = fast_sigmoid_lut(gate_inputs_ih[d] + gate_inputs_hh[d]);
                q8_7_t z = fast_sigmoid_lut(gate_inputs_ih[HIDDEN_DIM + d] + gate_inputs_hh[HIDDEN_DIM + d]);
                q8_7_t n = fast_tanh_lut(gate_inputs_ih[2 * HIDDEN_DIM + d] + (q8_7_t)(((int32_t)r * gate_inputs_hh[2 * HIDDEN_DIM + d]) >> 7));
                h[d] = (q8_7_t)(((int32_t)(FLOAT_TO_Q8_7(1.0f) - z) * n) >> 7) + (q8_7_t)(((int32_t)z * h[d]) >> 7);
            }
        }

        q8_7_t fc1_out[32] = {0};
        for(int i=0; i<32; ++i) {
            int32_t acc = 0;
            for(int j=0; j<HIDDEN_DIM; ++j) acc += ((int32_t)fc_policy_0_weight[i][j] * h[j]);
            fc1_out[i] = (q8_7_t)(acc >> 7) + fc_policy_0_bias[i];
            if(fc1_out[i] < 0) fc1_out[i] = 0;
        }

        q8_7_t max_logit = -32768;
        int target_beam_idx = 0;

        for (int i = 0; i < OUTPUT_DIM; ++i) {
            int32_t acc = 0;
            for (int j = 0; j < 32; ++j) acc += ((int32_t)fc_policy_2_weight[i][j] * fc1_out[j]);
            q8_7_t logit = (q8_7_t)(acc >> 7) + fc_policy_2_bias[i];
            if (logit > max_logit) {
                max_logit = logit;
                target_beam_idx = i;
            }
        }
        return target_beam_idx;
    }
};
#endif // WIRELESS_INFERENCE_ENGINE_HPP
