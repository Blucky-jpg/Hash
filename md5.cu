#include "md5.h"

__global__ void md5_bruteforce_Kernel(word **M, int numBlocks, word *T, word *output) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < numBlocks) {
        word *X = M[idx];
        
        word A = 0x67452301;
        word B = 0xefcdab89;
        word C = 0x98badcfe;
        word D = 0x10325476;
        
        word AA = A, BB = B, CC = C, DD = D;

        // Perform MD5 Rounds (R1_OP, R2_OP, R3_OP, R4_OP macros should be defined with CUDA-compatible operations)
        // Round 1
        R1_OP(A, B, C, D, X[0], 7, T[1]);
        R1_OP(D, A, B, C, X[1], 12, T[2]);
        R1_OP(C, D, A, B, X[2], 17, T[3]);
        R1_OP(B, C, D, A, X[3], 22, T[4]);

        R1_OP(A, B, C, D, X[4], 7, T[5]);
        R1_OP(D, A, B, C, X[5], 12, T[6]);
        R1_OP(C, D, A, B, X[6], 17, T[7]);
        R1_OP(B, C, D, A, X[7], 22, T[8]);

        R1_OP(A, B, C, D, X[8], 7, T[9]);
        R1_OP(D, A, B, C, X[9], 12, T[10]);
        R1_OP(C, D, A, B, X[10], 17, T[11]);
        R1_OP(B, C, D, A, X[11], 22, T[12]);

        R1_OP(A, B, C, D, X[12], 7, T[13]);
        R1_OP(D, A, B, C, X[13], 12, T[14]);
        R1_OP(C, D, A, B, X[14], 17, T[15]);
        R1_OP(B, C, D, A, X[15], 22, T[16]);

        // Round 2
        R2_OP(A, B, C, D, X[1], 5, T[17]);
        R2_OP(D, A, B, C, X[6], 9, T[18]);
        R2_OP(C, D, A, B, X[11], 14, T[19]);
        R2_OP(B, C, D, A, X[0], 20, T[20]);

        R2_OP(A, B, C, D, X[5], 5, T[21]);
        R2_OP(D, A, B, C, X[10], 9, T[22]);
        R2_OP(C, D, A, B, X[15], 14, T[23]);
        R2_OP(B, C, D, A, X[4], 20, T[24]);

        R2_OP(A, B, C, D, X[9], 5, T[25]);
        R2_OP(D, A, B, C, X[14], 9, T[26]);
        R2_OP(C, D, A, B, X[3], 14, T[27]);
        R2_OP(B, C, D, A, X[8], 20, T[28]);

        R2_OP(A, B, C, D, X[13], 5, T[29]);
        R2_OP(D, A, B, C, X[2], 9, T[30]);
        R2_OP(C, D, A, B, X[7], 14, T[31]);
        R2_OP(B, C, D, A, X[12], 20, T[32]);

        // Round 3
        R3_OP(A, B, C, D, X[5], 4, T[33]);
        R3_OP(D, A, B, C, X[8], 11, T[34]);
        R3_OP(C, D, A, B, X[11], 16, T[35]);
        R3_OP(B, C, D, A, X[14], 23, T[36]);

        R3_OP(A, B, C, D, X[1], 4, T[37]);
        R3_OP(D, A, B, C, X[4], 11, T[38]);
        R3_OP(C, D, A, B, X[7], 16, T[39]);
        R3_OP(B, C, D, A, X[10], 23, T[40]);

        R3_OP(A, B, C, D, X[13], 4, T[41]);
        R3_OP(D, A, B, C, X[0], 11, T[42]);
        R3_OP(C, D, A, B, X[3], 16, T[43]);
        R3_OP(B, C, D, A, X[6], 23, T[44]);

        R3_OP(A, B, C, D, X[9], 4, T[45]);
        R3_OP(D, A, B, C, X[12], 11, T[46]);
        R3_OP(C, D, A, B, X[15], 16, T[47]);
        R3_OP(B, C, D, A, X[2], 23, T[48]);

        // Round 4
        R4_OP(A, B, C, D, X[0], 6, T[49]);
        R4_OP(D, A, B, C, X[7], 10, T[50]);
        R4_OP(C, D, A, B, X[14], 15, T[51]);
        R4_OP(B, C, D, A, X[5], 21, T[52]);

        R4_OP(A, B, C, D, X[12], 6, T[53]);
        R4_OP(D, A, B, C, X[3], 10, T[54]);
        R4_OP(C, D, A, B, X[10], 15, T[55]);
        R4_OP(B, C, D, A, X[1], 21, T[56]);

        R4_OP(A, B, C, D, X[8], 6, T[57]);
        R4_OP(D, A, B, C, X[15], 10, T[58]);
        R4_OP(C, D, A, B, X[6], 15, T[59]);
        R4_OP(B, C, D, A, X[13], 21, T[60]);

        R4_OP(A, B, C, D, X[4], 6, T[61]);
        R4_OP(D, A, B, C, X[11], 10, T[62]);
        R4_OP(C, D, A, B, X[2], 15, T[63]);
        R4_OP(B, C, D, A, X[9], 21, T[64]);
        
        // Add AA, BB, CC, DD back into A, B, C, D
        A += AA;
        B += BB;
        C += CC;
        D += DD;

        // Store the result (you can copy it into global memory)
        output[idx * 4] = A;
        output[idx * 4 + 1] = B;
        output[idx * 4 + 2] = C;
        output[idx * 4 + 3] = D;
    }
}
