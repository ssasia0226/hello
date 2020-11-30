/////////////
#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <math.h>
#define FIXED_BIT 7  
#define int_BIT 5

void float2fix(float* n, int* m)
{
	unsigned int int_part = 0, frac_part = 0;
	int i;
	float k = *n;
	float t;
	if (k >= 32) {
		k = 32 - (1 / pow(2, FIXED_BIT));
	}
	if (k <= -32) {
		k = 32 - (1 / pow(2, FIXED_BIT));
		k = -k;
	}
	if (k > 0) {
		int_part = ((int)floor(k)) << FIXED_BIT;
		int_part &= (0b11111 << (FIXED_BIT));
	}
	else {
		k = -k;
		int_part = ((int)floor(k)) << FIXED_BIT; // need to round towards 0
		int_part = int_part & (0b11111 << FIXED_BIT);
		int_part |= (0b1 << (FIXED_BIT + int_BIT));
	}
	k = fabs(k) - floor(fabs(k));

	t = 0.5;
	for (i = 0; i < FIXED_BIT; i++) {
		if ((k - t) >= 0) {
			k -= t;
			frac_part += (1 << (FIXED_BIT - 1 - i));
		}
		t = t / 2;
	}
	*m = int_part + frac_part;
}
void fix2float(int* n, float* m) {
	float num;
	int k = *n;
	if ((k >> (FIXED_BIT + int_BIT)) & 0b1) {
		k ^= (1 << (FIXED_BIT + int_BIT));
		num = -k / pow(2, FIXED_BIT);
	}
	else num = k / pow(2, FIXED_BIT);
	*m = num;
}