#include "puncture.h"

bool matrix_1_2[] = {1,1};
bool matrix_2_3[] = {1,1,1,0};
bool matrix_3_4[] = {1,1,1,0,0,1};
bool matrix_5_6[] = {1,1,1,0,0,1,1,0,0,1};
bool matrix_7_8[] = {1,1,1,0,1,0,1,0,0,1,1,0,0,1};
bool matrix_1_1[] = {0,1,1,0};
bool matrix_3_2[] = {0,0,1,0,1,0};
bool matrix_2_1[] = {0,0,0,1,0,0,1,0};

#define ASSIGN_VECTOR(target, source) target.assign(source, source+sizeof(source)/sizeof(source[0]))

PuncturingMask::PuncturingMask(int numerator, int denominator)
    : numerator(numerator), denominator(denominator)
{
    if (numerator == 1 && denominator == 2)
        ASSIGN_VECTOR(mask, matrix_1_2);
    else if (numerator == 2 && denominator == 3)
        ASSIGN_VECTOR(mask, matrix_2_3);
    else if (numerator == 3 && denominator == 4)
        ASSIGN_VECTOR(mask, matrix_3_4);
    else if (numerator == 5 && denominator == 6)
        ASSIGN_VECTOR(mask, matrix_5_6);
    else if (numerator == 7 && denominator == 8)
        ASSIGN_VECTOR(mask, matrix_7_8);
    else if (numerator == 1 && denominator == 1)
        ASSIGN_VECTOR(mask, matrix_1_1);
    else if (numerator == 3 && denominator == 2)
        ASSIGN_VECTOR(mask, matrix_3_2);
    else if (numerator == 2 && denominator == 1)
        ASSIGN_VECTOR(mask, matrix_2_1);
}

void PuncturingMask::puncture(const bitvector &input, bitvector &output) const {
    int count = 0;
    int N = input.size(), M = mask.size();
    for (int i=0; i<N; i++)
        count += mask[i % M];
    output.resize(count);
    int j=0;
    for (int i=0; i<N; i++)
        if (mask[i % M])
            output[j++] = input[i];
}

void PuncturingMask::depuncture(const std::vector<int> &input, std::vector<int> &output) const {
    int N = input.size(), M = mask.size();
    int m = 0;
    for (int i=0; i<M; i++)
        m += mask[i];
    int L = ((N + m - 1) / m) * M;
    output.resize(L);
    int j = 0;
    for (int i=0; i<L; i++)
        output[i] = (mask[i % M]) ? input[j++] : 0;
}
