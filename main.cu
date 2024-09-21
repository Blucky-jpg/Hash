#include "md5.h"
#include <stdio.h>
#include <string.h>
#include <stdbool.h>
#include <time.h>
#include <pthread.h>
#include <unistd.h>
#include <cuda_runtime.h>
#include <math.h>
#include <atomic>

#define NUM_TARGET_HASHES 13
#define MAX_LEN 5
#define BLOCK_SIZE 256

// Target hashes, in the same format as before
const char *targetHashes[NUM_TARGET_HASHES] = {
"8d2855c1ed9047f655ce86f58ee3c4ed", "e3b0cf89846e85460d5f11caa2384acb",
"a7fb77d397a34877ab875e71f4bf236d", "683c11c14c7cb33c58c6220b5c1c7f51",
"3406e8d8f5244aeda183a1a71fbc2be6", "7013744587af48c686824a8b5015b075",
"9cf8e0a3a32e41039410f88f4ff101af", "c21483dc4ddc40a2a9c2a873af0b8f8c",
"175e915e8a5b4bcca345ee75b568ee63", "5c098bca9cb8464687d78c580b6a4a78",
"748ce7269dfd461787f3513d6d6e5352", "ae6fdb1ebdf7445eb0cd0ff9d111d34c",
"4ae57592275241e4912bc106ad470b04"
};

std::atomic<size_t> hashCount(0);  // Atomic counter for thread safety

// Check if the generated hash matches any target hash
__host__ bool isTargetHash(char *generatedHash) {
    for (int i = 0; i < NUM_TARGET_HASHES; i++) {
        if (strcmp(generatedHash, targetHashes[i]) == 0) {
            return true;
        }
    }
    return false;
}

// CUDA Error Check Macro
#define cudaCheckError() { \
    cudaError_t e = cudaGetLastError(); \
    if (e != cudaSuccess) { \
        printf("CUDA Error: %s\n", cudaGetErrorString(e)); \
        exit(EXIT_FAILURE); \
    } \
}

// Host MD5 brute-force function using CUDA
void bruteForceMD5_CUDA(char *charset, int maxLen) {
    int charsetSize = strlen(charset);
    
   // Allocate GPU memory and copy data
    char *d_charset;
    cudaMalloc((void **)&d_charset, charsetSize * sizeof(char));
    cudaCheckError();
    cudaMemcpy(d_charset, charset, charsetSize * sizeof(char), cudaMemcpyHostToDevice);
    cudaCheckError();

    // Allocate other memory
    word **d_M;
    word *d_output;
    word T[65];  // MD5 T-table
    cudaDeviceSetLimit(cudaLimitMallocHeapSize, 1024 * 1024 * 1024);


    cudaMalloc((void **)&d_M, sizeof(word *) * BLOCK_SIZE);
    cudaCheckError();
    cudaMalloc((void **)&d_M, sizeof(word *) * BLOCK_SIZE);
    cudaCheckError();  // Ensure allocation is successful
    word *d_T;
    cudaMalloc((void **)&d_T, 65 * sizeof(word));
    cudaCheckError();
    cudaMemcpy(d_T, T, 65 * sizeof(word), cudaMemcpyHostToDevice);
    cudaCheckError();
    
    // Compute the total number of combinations
    int totalCombinations = pow(charsetSize, maxLen);
    int numBlocks = (totalCombinations + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // Launch kernel
    md5_bruteforce_Kernel<<<numBlocks, BLOCK_SIZE>>>(d_M, numBlocks, d_T, d_output);
    cudaDeviceSynchronize();  // Ensure kernel execution is complete
    cudaCheckError();  // Check for errors after kernel execution

    // Copy output back to host
    word *output = (word *)malloc(sizeof(word) * 4 * BLOCK_SIZE);
    cudaMemcpy(output, d_output, sizeof(word) * 4 * BLOCK_SIZE, cudaMemcpyDeviceToHost);
    cudaCheckError();

    // Process results
    for (int i = 0; i < numBlocks * BLOCK_SIZE; i++) {
        char generatedHash[33];
        snprintf(generatedHash, sizeof(generatedHash), "%08x%08x%08x%08x", output[i * 4], output[i * 4 + 1], output[i * 4 + 2], output[i * 4 + 3]);
        
        if (isTargetHash(generatedHash)) {
            printf("Found match: %s\n", generatedHash);
        }
        hashCount.fetch_add(1);
    }

    // Free memory
    cudaFree(d_charset);
    cudaFree(d_M);
    cudaFree(d_output);
    cudaFree(d_T);
    free(output);
}

// Stats printing function running in a separate thread
void *printStats(void *arg) {
    size_t lastHashCount = 0;
    time_t lastTime = time(NULL); // Start the timer

    while (1) {
        sleep(5);  // Sleep for 5 seconds
        time_t currentTime = time(NULL);
        double elapsedTime = difftime(currentTime, lastTime);

        size_t currentHashCount = hashCount.load();
        size_t hashesProcessed = currentHashCount - lastHashCount;
        double hashesPerSecond = hashesProcessed / elapsedTime;
        printf("Hashes per second: %.2f\n", hashesPerSecond);

        // Update last values
        lastHashCount = currentHashCount;
        lastTime = currentTime;
    }

    return NULL; // Required for pthread functions
}

int main() {
    // Charset and maximum length for brute-forcing
    char charset[] = "abcdefghijklmnopqrstuvwxyz0123456789";
    int maxLen = MAX_LEN;

    // Start the stats printing in a separate thread
    pthread_t statsThread;
    pthread_create(&statsThread, NULL, printStats, NULL);

    // Start brute-forcing using CUDA
    bruteForceMD5_CUDA(charset, maxLen);

    // Wait for the stats thread to finish (it won't, but this is a clean way to join threads)
    pthread_join(statsThread, NULL);

    return 0;
}
