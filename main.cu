#include <stdio.h>
#include <cuda.h>
#include <openssl/md5.h>

#define MAX_PASSWORD_LENGTH 6
#define CHARSET_SIZE 62 // a-z, A-Z, 0-9
#define HASH_COUNT 13

__device__ void md5_hash(const char *input, unsigned char *output) {
    MD5((unsigned char*)input, strlen(input), output);
}

__global__ void crack_hashes(const char *target_hashes, int *found, char *result) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    char password[MAX_PASSWORD_LENGTH + 1];
    unsigned char hash[MD5_DIGEST_LENGTH];

    // Brute-force logic: generate passwords from idx
    for (int i = 0; i < CHARSET_SIZE; i++) {
        for (int j = 0; j < CHARSET_SIZE; j++) {
            for (int k = 0; k < CHARSET_SIZE; k++) {
                for (int l = 0; l < CHARSET_SIZE; l++) {
                    for (int m = 0; m < CHARSET_SIZE; m++) {
                        for (int n = 0; n < CHARSET_SIZE; n++) {
                            // Generate password
                            password[0] = 'a' + (i % 26);
                            password[1] = 'a' + (j % 26);
                            password[2] = 'a' + (k % 26);
                            password[3] = 'a' + (l % 26);
                            password[4] = 'a' + (m % 26);
                            password[5] = 'a' + (n % 26);
                            password[6] = '\0';

                            // Hash the password
                            md5_hash(password, hash);

                            // Compare with target hashes
                            for (int t = 0; t < HASH_COUNT; t++) {
                                if (memcmp(hash, &target_hashes[t * MD5_DIGEST_LENGTH], MD5_DIGEST_LENGTH) == 0) {
                                    found[t] = 1; // Found a match
                                    strcpy(&result[t * (MAX_PASSWORD_LENGTH + 1)], password); // Store the result
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

void launch_kernel(const char *target_hashes) {
    char *d_target_hashes;
    int *d_found;
    char *d_result;

    // Allocate device memory
    cudaMalloc(&d_target_hashes, HASH_COUNT * MD5_DIGEST_LENGTH);
    cudaMalloc(&d_found, HASH_COUNT * sizeof(int));
    cudaMalloc(&d_result, HASH_COUNT * (MAX_PASSWORD_LENGTH + 1));

    cudaMemcpy(d_target_hashes, target_hashes, HASH_COUNT * MD5_DIGEST_LENGTH, cudaMemcpyHostToDevice);
    cudaMemset(d_found, 0, HASH_COUNT * sizeof(int));

    int threads_per_block = 256;
    int blocks = (CHARSET_SIZE * CHARSET_SIZE * CHARSET_SIZE * CHARSET_SIZE * CHARSET_SIZE * CHARSET_SIZE + threads_per_block - 1) / threads_per_block;

    // Launch the kernel
    crack_hashes<<<blocks, threads_per_block>>>(d_target_hashes, d_found, d_result);

    // Copy results back to host
    int found[HASH_COUNT];
    char result[HASH_COUNT][MAX_PASSWORD_LENGTH + 1];
    cudaMemcpy(found, d_found, HASH_COUNT * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(result, d_result, HASH_COUNT * (MAX_PASSWORD_LENGTH + 1), cudaMemcpyDeviceToHost);

    // Print results
    for (int i = 0; i < HASH_COUNT; i++) {
        if (found[i]) {
            printf("Found password for hash %d: %s\n", i, result[i]);
        }
    }

    // Free device memory
    cudaFree(d_target_hashes);
    cudaFree(d_found);
    cudaFree(d_result);
}

int main() {
    // Example target hashes (replace with actual MD5 hashes)
    const char target_hashes[HASH_COUNT][MD5_DIGEST_LENGTH] = {
        // Fill with your target MD5 hashes
    };

    launch_kernel((const char *)target_hashes);

    return 0;
}
