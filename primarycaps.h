#pragma once
#include<math.h>
typedef struct PrimaryCaps_layer {
    float kernel[8 * 32 * 256 * 9 * 9];
    float bias[8 * 32];
    float output[8 * 32 * 6 * 6];
    float sq[8 * 1152];
} priLayer;
void conv_pri(float input[256 * 20 * 20], float kernel[8 * 32 * 256 * 9 * 9], float bias[8 * 32], float output[8 * 32 * 6 * 6]) {
    int i, j, k, l, m, n, o, p;
    for (o = 0; o < 8; o++) {
        for (i = 0; i < 6; i++) {
            for (j = 0; j < 6; j++) {
                for (k = 0; k < 32; k++) {
                    output[o * 32 * 6 * 6 + k * 6 * 6 + i * 6 + j] = bias[o * 32 + k];
                    for (p = 0; p < 256; p++) {
                        for (l = 0; l < 9; l++) {
                            for (m = 0; m < 9; m++) {
                                output[o * 32 * 6 * 6 + k * 6 * 6 + i * 6 + j] += input[p * 20 * 20 + (2 * i + l) * 20 + 2 * j + m] * kernel[o * 32 * 9 * 9 * 256 + k * 9 * 9 * 256 + p * 9 * 9 + l * 9 + m];
                            }
                        }
                    }
                }
            }
        }
    }
}


void squash(float input[8 * 1152], float output[8 * 1152]) {
    int i, j, k;
    float tmp = 0;
    for (i = 0; i < 8; i++) {
        for (j = 0; j < 1152; j++) {
            for (k = 0; k < 1152; k++) {
                tmp += pow(input[k + 1152 * i], 2);
            }
            output[i * 1152 + j] = input[i * 1152 + j] * tmp / (tmp + 1) / sqrt(tmp);
            tmp = 0;
        }
    }
}
