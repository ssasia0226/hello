
extern float t_output[1152 * 8];
extern float c_ij[11520];
extern float s_j1[1152 * 10 * 16];
extern float v_j[160];
extern float s_j[160];
extern float v_j1[1152 * 10 * 16];
extern float u_vj1[1152 * 10];
extern float b_ij[11520];
extern float u_hat[1152 * 10 * 16];

typedef struct digits_layer {
    float W[1152 * 10 * 16 * 8];
    float u_hat[1152 * 10 * 16];
    float v_j[10 * 160];
    float b_ij[11520];
} digitcapsLayer;

float softmax(float n, float sum) {
    float result;
    result = exp(n) / sum;
    return result;
}

void u_hat_maker(float input[8 * 1152], float W[1152 * 10 * 16 * 8]) {
    float t_output[8 * 1152];
    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 1152; j++) {
            t_output[j * 8 + i] = 0;
            t_output[j * 8 + i] = input[i * 1152 + j];
        }
    }
    float s_output[1152 * 10 * 8];
    for (int i = 0; i < 1152; i++) {
        for (int j = 0; j < 10; j++) {
            for (int k = 0; k < 8; k++) {
                s_output[i * 10 * 8 + j * 8 + k] = 0;
                s_output[i * 10 * 8 + j * 8 + k] = t_output[i * 8 + k];
            }
        }
    }
    for (int i = 0; i < 11520; i++) {
        b_ij[i] = 0;
    }
    for (int i = 0; i < 1152 * 10 * 16; i++) {
        u_hat[i] = 0;
    }
    for (int l = 0; l < 1152; l++) { // matmul
        for (int k = 0; k < 10; k++) {
            for (int j = 0; j < 16; j++) {
                for (int i = 0; i < 8; i++) {
                    u_hat[j + 16 * k + 16 * 10 * l] += W[l * 16 * 8 * 10 + 16 * 8 * k + 8 * j + i] * s_output[l * 8 * 10 + k * 8 + i];
                }
            }
        }
    }// U_hat 결과 >>> u_hat[1152 * 10 * 16 * 1]
}

void iteration_1() { // [ 1152 * 8 ] * W 결과를 인풋으로 받음
    for (int i = 0; i < 11520; i++) {
        c_ij[i] = 0;
    }
    for (int i = 0; i < 1 * 1152 * 10 * 16 * 1; i++) {
        s_j1[i] = 0;
    }
    for (int i = 0; i < 160; i++) {
        s_j[i] = 0;
    }
    for (int i = 0; i < 160; i++) {
        v_j[i] = 0;
    }
    for (int i = 0; i < 1 * 1152 * 10 * 16 * 1; i++) {
        v_j1[i] = 0;
    }
    for (int i = 0; i < 1 * 1152 * 10 * 1; i++) {
        u_vj1[i] = 0;
    }

    // routing iteration 3회
    float sum[10];
    for (int i = 0; i < 10; i++) sum[i] = 0;
    for (int j = 0; j < 10; j++) {
        for (int i = 0; i < 1152; i++) {
            sum[j] += exp(b_ij[j + 10 * i]);
        }
    }
    for (int j = 0; j < 10; j++) {
        for (int i = 0; i < 1152; i++) {
            c_ij[j + 10 * i] = softmax(b_ij[j + 10 * i], sum[j]);
        }
    }// softmax결과 차원 >>> [1152 * 10 * 1 * 1]

    for (int k = 0; k < 1152; k++) {
        for (int j = 0; j < 10; j++) {
            for (int i = 0; i < 16; i++) {
                s_j1[i + j * 16 + 10 * 16 * k] += c_ij[10 * k + j] * u_hat[10 * 16 * k + 16 * j + i];
            }
        }
    }

    for (int j = 0; j < 10; j++) {
        for (int i = 0; i < 16; i++) {
            for (int k = 0; k < 1152; k++) {
                s_j[i + j * 16] += s_j1[i + j * 16 + 10 * 16 * k];
            }
        }
    }

    /*squash*/
    float tmp = 0;
    for (int i = 0; i < 16; i++) {
        for (int j = 0; j < 10; j++) {
            for (int k = 0; k < 10; k++) {
                tmp += pow(s_j[k * 16 + i], 2);
            }
            v_j[i + j * 16] = s_j[i + j * 16] * tmp / (tmp + 1) / sqrt(tmp);
            tmp = 0;
        }
    }

    for (int k = 0; k < 1152; k++) {
        for (int j = 0; j < 10; j++) {
            for (int i = 0; i < 16; i++) {
                v_j1[k * 10 * 16 + j * 16 + i] = v_j[16 * j + i];
            }
        }
    }

    for (int k = 0; k < 1152; k++) {
        for (int j = 0; j < 10; j++) {
            for (int i = 0; i < 16; i++) {
                u_vj1[k * 10 + j] += u_hat[k * 10 * 16 + j * 16 + i] * v_j1[k * 10 * 16 + j * 16 + i];
            }
        }
    }

    for (int i = 0; i < 1152; i++) {
        for (int j = 0; j < 10; j++) {
            b_ij[10 * i + j] = b_ij[10 * i + j] + u_vj1[10 * i + j];
        }
    }

    /*세번째 iteration*/
}
void iteration_2() {
    /*2번째 iteration*/
    for (int i = 0; i < 11520; i++) {
        c_ij[i] = 0;
    }
    for (int i = 0; i < 1 * 1152 * 10 * 16 * 1; i++) {
        s_j1[i] = 0;
    }
    for (int i = 0; i < 160; i++) {
        s_j[i] = 0;
    }
    for (int i = 0; i < 160; i++) {
        v_j[i] = 0;
    }
    for (int i = 0; i < 1 * 1152 * 10 * 16 * 1; i++) {
        v_j1[i] = 0;
    }
    for (int i = 0; i < 1 * 1152 * 10 * 1; i++) {
        u_vj1[i] = 0;
    }
    // routing iteration 3회
    float sum[10];
    for (int i = 0; i < 10; i++) sum[i] = 0;
    for (int j = 0; j < 10; j++) {
        for (int i = 0; i < 1152; i++) {
            sum[j] += exp(b_ij[j + 10 * i]);
        }
    }
    for (int j = 0; j < 10; j++) {
        for (int i = 0; i < 1152; i++) {
            c_ij[j + 10 * i] = softmax(b_ij[j + 10 * i], sum[j]);
        }
    }// softmax결과 차원 >>> [1152 * 10 * 1 * 1]


    for (int k = 0; k < 1152; k++) {
        for (int j = 0; j < 10; j++) {
            for (int i = 0; i < 16; i++) {
                s_j1[i + j * 16 + 10 * 16 * k] += c_ij[10 * k + j] * u_hat[10 * 16 * k + 16 * j + i];
            }
        }
    }

    for (int j = 0; j < 10; j++) {
        for (int i = 0; i < 16; i++) {
            for (int k = 0; k < 1152; k++) {
                s_j[i + j * 16] += s_j1[i + j * 16 + 10 * 16 * k];
            }
        }
    }


    /*squash*/
    float tmp = 0;
    for (int i = 0; i < 16; i++) {
        for (int j = 0; j < 10; j++) {
            for (int k = 0; k < 10; k++) {
                tmp += pow(s_j[k * 16 + i], 2);
            }
            v_j[i + j * 16] = s_j[i + j * 16] * tmp / (tmp + 1) / sqrt(tmp);
            tmp = 0;
        }
    }
    for (int k = 0; k < 1152; k++) {
        for (int j = 0; j < 10; j++) {
            for (int i = 0; i < 16; i++) {
                v_j1[k * 10 * 16 + j * 16 + i] = v_j[16 * j + i];
            }
        }
    }

    for (int k = 0; k < 1152; k++) {
        for (int j = 0; j < 10; j++) {
            for (int i = 0; i < 16; i++) {
                u_vj1[k * 10 + j] += u_hat[k * 10 * 16 + j * 16 + i] * v_j1[k * 10 * 16 + j * 16 + i];
            }
        }
    }

    for (int i = 0; i < 1152; i++) {
        for (int j = 0; j < 10; j++) {
            b_ij[10 * i + j] = b_ij[10 * i + j] + u_vj1[10 * i + j];
        }
    }

}

void iteration_3(float a[160 * 10], int data) {
    /*2번째 iteration*/
    for (int i = 0; i < 11520; i++) {
        c_ij[i] = 0;
    }
    for (int i = 0; i < 1 * 1152 * 10 * 16 * 1; i++) {
        s_j1[i] = 0;
    }
    for (int i = 0; i < 160; i++) {
        s_j[i] = 0;
    }
    for (int i = 0; i < 160; i++) {
        v_j[i] = 0;
    }
    for (int i = 0; i < 1 * 1152 * 10 * 16 * 1; i++) {
        v_j1[i] = 0;
    }
    for (int i = 0; i < 1 * 1152 * 10 * 1; i++) {
        u_vj1[i] = 0;
    }
    // routing iteration 3회
    float sum[10];
    for (int i = 0; i < 10; i++) sum[i] = 0;
    for (int j = 0; j < 10; j++) {
        for (int i = 0; i < 1152; i++) {
            sum[j] += exp(b_ij[j + 10 * i]);
        }
    }
    for (int j = 0; j < 10; j++) {
        for (int i = 0; i < 1152; i++) {
            c_ij[j + 10 * i] = softmax(b_ij[j + 10 * i], sum[j]);
        }
    }// softmax결과 차원 >>> [1152 * 10 * 1 * 1]


    for (int k = 0; k < 1152; k++) {
        for (int j = 0; j < 10; j++) {
            for (int i = 0; i < 16; i++) {
                s_j1[i + j * 16 + 10 * 16 * k] += c_ij[10 * k + j] * u_hat[10 * 16 * k + 16 * j + i];
            }
        }
    }

    for (int j = 0; j < 10; j++) {
        for (int i = 0; i < 16; i++) {
            for (int k = 0; k < 1152; k++) {
                s_j[i + j * 16] += s_j1[i + j * 16 + 10 * 16 * k];
            }
        }
    }


    /*squash*/
    float tmp = 0;
    for (int i = 0; i < 16; i++) {
        for (int j = 0; j < 10; j++) {
            for (int k = 0; k < 10; k++) {
                tmp += pow(s_j[k * 16 + i], 2);
            }
            v_j[i + j * 16] = s_j[i + j * 16] * tmp / (tmp + 1) / sqrt(tmp);
            a[data * 160 + i + j * 16] = v_j[i + j * 16];
            tmp = 0;
        }
    }
}