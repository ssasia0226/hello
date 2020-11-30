#pragma once
typedef struct convolutional_layer {
    float input_image[28 * 28];      //input_image[28][28]
    float kernel[256 * 9 * 9];        //kernel[256][9][9]
    float output_1[256 * 20 * 20];    //output_1[256][20][20]
    float output[256 * 20 * 20];      //output[256][20][20]
    float bias[256];
} convLayer;

void ReLu(float o[256 * 20 * 20], float output[256 * 20 * 20]) {            //float o[256][20][20], float output[256][20][20]
    int i, j, k;
    for (i = 0; i < 256 * 20 * 20; i++) {
        if (o[i] >= 0) output[i] = o[i];      //if (o[i][j][k] >= 0) output[i][j][k] = o[i][j][k];
        else output[i] = 0;                                             //else output[i][j][k] = 0;
    }
}
void convolution(float input[28 * 28], float kernel[256 * 9 * 9], float bias[256], float output[256 * 20 * 20]) {
    //float input[28][28], float kernel[256][9][9], float bias[256], float output[256][20][20]
    int i, j, k, l, m;
    for (i = 0; i < 20; i++) {
        for (j = 0; j < 20; j++) {
            for (k = 0; k < 256; k++) {
                output[k * 20 * 20 + i * 20 + j] = bias[k];          //output[k][i][j] = bias[k];
                for (l = 0; l < 9; l++) {
                    for (m = 0; m < 9; m++) {
                        output[k * 20 * 20 + i * 20 + j] += input[(i + l) * 28 + j + m] * kernel[k * 81 + l * 9 + m];       //output[k][i][j] += input[i + l][j + m] * kernel[k][l][m];
                    }
                }
            }
        }
    }
}