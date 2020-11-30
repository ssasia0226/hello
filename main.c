/////////////
#define _CRT_SECURE_NO_WARNINGS


#include "io.h"
#include "system.h"
#include "npy_array.h"
#include "zipcontainer.h"
#include "crc32.h"
#include "dostime.h"
#include <stdbool.h>
#include <math.h>
#include "conv0.h"
#include "primarycaps.h"
#include "digitcaps.h"
#include "fixedpoint.h"
#include <stdio.h>
#include <stdlib.h>
float t_output[1152 * 8];
float c_ij[11520];
float s_j1[1152 * 10 * 16];
float v_j[160];
float s_j[160];
float v_j1[1152 * 10 * 16];
float u_vj1[1152 * 10];
float b_ij[11520];
float u_hat[1152 * 10 * 16];
float v_j1_fix[1152 * 10 * 16];

int main() {
	convLayer* conv1 = (convLayer*)malloc(sizeof(convLayer));
	priLayer* pri = (priLayer*)malloc(sizeof(priLayer));
	digitcapsLayer* digits = (digitcapsLayer*)malloc(sizeof(digitcapsLayer));
	
	char* mnist = "mnist_1000.npy";
	char* bias0 = "conv0.bias.npy";
	char* weight0 = "conv0.weight.npy";
	char* unit0bias = "unit0_conv0.bias.npy";
	char* unit0weight = "unit0_conv0.weight.npy";
	char* unit1bias = "unit1_conv0.bias.npy";
	char* unit1weight = "unit1_conv0.weight.npy";
	char* unit2bias = "unit2_conv0.bias.npy";
	char* unit2weight = "unit2_conv0.weight.npy";
	char* unit3bias = "unit3_conv0.bias.npy";
	char* unit3weight = "unit3_conv0.weight.npy";
	char* unit4bias = "unit4_conv0.bias.npy";
	char* unit4weight = "unit4_conv0.weight.npy";
	char* unit5bias = "unit5_conv0.bias.npy";
	char* unit5weight = "unit5_conv0.weight.npy";
	char* unit6bias = "unit6_conv0.bias.npy";
	char* unit6weight = "unit6_conv0.weight.npy";
	char* unit7bias = "unit7_conv0.bias.npy";
	char* unit7weight = "unit7_conv0.weight.npy";
	char* weight = "W.npy";
	char* result = "result.npy";
	npy_array_t* result7 = npy_array_load(result);
	npy_array_dump(result7);
	npy_array_t* mnist8 = npy_array_load(mnist);
	npy_array_dump(mnist8);

	npy_array_t* conv0_bias = npy_array_load(bias0);
	npy_array_dump(conv0_bias);
	npy_array_t* conv0_weight = npy_array_load(weight0);
	npy_array_dump(conv0_weight);

	npy_array_t* unit0_bias = npy_array_load(unit0bias);
	npy_array_dump(unit0_bias);
	npy_array_t* unit0_weight = npy_array_load(unit0weight);
	npy_array_dump(unit0_weight);

	npy_array_t* unit1_bias = npy_array_load(unit1bias);
	npy_array_dump(unit1_bias);
	npy_array_t* unit1_weight = npy_array_load(unit1weight);
	npy_array_dump(unit1_weight);

	npy_array_t* unit2_bias = npy_array_load(unit2bias);
	npy_array_dump(unit2_bias);
	npy_array_t* unit2_weight = npy_array_load(unit2weight);
	npy_array_dump(unit2_weight);

	npy_array_t* unit3_bias = npy_array_load(unit3bias);
	npy_array_dump(unit3_bias);
	npy_array_t* unit3_weight = npy_array_load(unit3weight);
	npy_array_dump(unit3_weight);

	npy_array_t* unit4_bias = npy_array_load(unit4bias);
	npy_array_dump(unit4_bias);
	npy_array_t* unit4_weight = npy_array_load(unit4weight);
	npy_array_dump(unit4_weight);

	npy_array_t* unit5_bias = npy_array_load(unit5bias);
	npy_array_dump(unit5_bias);
	npy_array_t* unit5_weight = npy_array_load(unit5weight);
	npy_array_dump(unit5_weight);

	npy_array_t* unit6_bias = npy_array_load(unit6bias);
	npy_array_dump(unit6_bias);
	npy_array_t* unit6_weight = npy_array_load(unit6weight);
	npy_array_dump(unit6_weight);

	npy_array_t* unit7_bias = npy_array_load(unit7bias);
	npy_array_dump(unit7_bias);
	npy_array_t* unit7_weight = npy_array_load(unit7weight);
	npy_array_dump(unit7_weight);

	npy_array_t* W = npy_array_load(weight);
	npy_array_dump(W);
	//printf("------------------------------Pytorch RESULT----------------------------------------\n");
	/*for(int k=0; k<10; k++){
		for (int i = 0; i < 10; i++) {
			for (int j = 0; j < 16; j++) {
				printf("%f, ", result7->data[k*160+j + i * 16]);
			}
		printf("\n");
		}
	}*/
	int i;
	for (i = 0; i < 256; i++) {
		conv1->bias[i] = conv0_bias->data[i];
	}
	for (i = 0; i < 256 * 9 * 9; i++) {
		conv1->kernel[i] = conv0_weight->data[i];
	}
	for (i = 0; i < 32; i++) {
		pri->bias[i] = unit0_bias->data[i];
	}
	for (i = 0; i < 32; i++) {
		pri->bias[32 + i] = unit1_bias->data[i];
	}
	for (i = 0; i < 32; i++) {
		pri->bias[32 * 2 + i] = unit2_bias->data[i];
	}
	for (i = 0; i < 32; i++) {
		pri->bias[32 * 3 + i] = unit3_bias->data[i];
	}
	for (i = 0; i < 32; i++) {
		pri->bias[32 * 4 + i] = unit4_bias->data[i];
	}
	for (i = 0; i < 32; i++) {
		pri->bias[32 * 5 + i] = unit5_bias->data[i];
	}
	for (i = 0; i < 32; i++) {
		pri->bias[32 * 6 + i] = unit6_bias->data[i];
	}
	for (i = 0; i < 32; i++) {
		pri->bias[32 * 7 + i] = unit7_bias->data[i];
	}
	for (i = 0; i < 32 * 256 * 9 * 9; i++) {
		pri->kernel[i] = unit0_weight->data[i];
	}
	for (i = 0; i < 32 * 256 * 9 * 9; i++) {
		pri->kernel[i + 32 * 256 * 9 * 9] = unit1_weight->data[i];
	}
	for (i = 0; i < 32 * 256 * 9 * 9; i++) {
		pri->kernel[i + 32 * 256 * 9 * 9 * 2] = unit2_weight->data[i];
	}
	for (i = 0; i < 32 * 256 * 9 * 9; i++) {
		pri->kernel[i + 32 * 256 * 9 * 9 * 3] = unit3_weight->data[i];
	}
	for (i = 0; i < 32 * 256 * 9 * 9; i++) {
		pri->kernel[i + 32 * 256 * 9 * 9 * 4] = unit4_weight->data[i];
	}
	for (i = 0; i < 32 * 256 * 9 * 9; i++) {
		pri->kernel[i + 32 * 256 * 9 * 9 * 5] = unit5_weight->data[i];
	}
	for (i = 0; i < 32 * 256 * 9 * 9; i++) {
		pri->kernel[i + 32 * 256 * 9 * 9 * 6] = unit6_weight->data[i];
	}
	for (i = 0; i < 32 * 256 * 9 * 9; i++) {
		pri->kernel[i + 32 * 256 * 9 * 9 * 7] = unit7_weight->data[i];
	}
	for (i = 0; i < 1152 * 10 * 16 * 8; i++) {
		digits->W[i] = W->data[i];
	}
	for (i = 0; i < 10; i++) {
		convolution(mnist8->data + i * 784, conv1->kernel, conv1->bias, conv1->output_1);
		ReLu(conv1->output_1, conv1->output);
		conv_pri(conv1->output, pri->kernel, pri->bias, pri->output);
		squash(pri->output, pri->sq);
		u_hat_maker(pri->sq, digits->W);
		iteration_1();
		iteration_2();
		iteration_3(digits->v_j, i);
		printf("image%d complete.\n", i + 1);
	}

	printf("\n------------------------------C RESULT----------------------------------------\n");
	for (int k = 0; k < 10; k++) {
		for (int i = 0; i < 10; i++) {
			for (int j = 0; j < 16; j++) {
				printf("%f, ", digits->v_j[k * 160 + j + i * 16]);
			}
			printf("\n");
		}
		printf("\n------------------------------image %d----------------------------------------\n", k + 1);
	}

	

	return 0;
}
