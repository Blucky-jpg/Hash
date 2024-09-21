#include "md5.h"
#include <stdio.h>
#include <string.h>
#include <stdbool.h>
#include <time.h>
#include <pthread.h>
#include <unistd.h>

#define NUM_TARGET_HASHES 13

// Target hashes
const char *targetHashes[NUM_TARGET_HASHES] = {
    "8d2855c1ed9047f655ce86f58ee3c4ed",
    "e3b0cf89846e85460d5f11caa2384acb",
    "a7fb77d397a34877ab875e71f4bf236d",
    "683c11c14c7cb33c58c6220b5c1c7f51",
    "3406e8d8f5244aeda183a1a71fbc2be6",
    "7013744587af48c686824a8b5015b075",
    "9cf8e0a3a32e41039410f88f4ff101af",
    "c21483dc4ddc40a2a9c2a873af0b8f8c",
    "175e915e8a5b4bcca345ee75b568ee63",
    "5c098bca9cb8464687d78c580b6a4a78",
    "748ce7269dfd461787f3513d6d6e5352",
    "ae6fdb1ebdf7445eb0cd0ff9d111d34c",
    "4ae57592275241e4912bc106ad470b04"
};

// Prototype for generateCombinations
void generateCombinations(char *charset, char *attempt, int position, int maxLen, int charsetSize);

size_t hashCount = 0;  // Counter to track the number of hashes

// Function to compare generated hash with target hashes
bool isTargetHash(char *generatedHash) {
    for (int i = 0; i < NUM_TARGET_HASHES; i++) {
        if (strcmp(generatedHash, targetHashes[i]) == 0) {
            return true;
        }
    }
    return false;
}

// Brute-force function (you can modify the charset and maxLen)
void bruteForceMD5(char *charset, int maxLen) {
    int charsetSize = strlen(charset);
    char attempt[maxLen + 1];
    memset(attempt, 0, maxLen + 1);

    // Recursive function to generate combinations
    for (int len = 1; len <= maxLen; len++) {
        for (int i = 0; i < charsetSize; i++) {
            attempt[0] = charset[i];
            generateCombinations(charset, attempt, 1, len, charsetSize);
        }
    }
}

// Recursive function to generate combinations
void generateCombinations(char *charset, char *attempt, int position, int maxLen, int charsetSize) {
    if (position == maxLen) {
        Blocks *blocks = makeBlocks((ubyte *)attempt, strlen(attempt));
        char *generatedHash = md5(blocks);
        
        if (isTargetHash(generatedHash)) {
            printf("Found match: %s -> %s\n", attempt, generatedHash);
        }

        free(generatedHash);
        hashCount++;  // Increment hash count for each hash generated
        return;
    }

    for (int i = 0; i < charsetSize; i++) {
        attempt[position] = charset[i];
        generateCombinations(charset, attempt, position + 1, maxLen, charsetSize);
    }
}

// Stats printing function running in a separate thread
void *printStats(void *arg) {
    size_t *hashCount = (size_t *)arg; // Cast back to size_t*

    time_t startTime = time(NULL); // Start the timer
    time_t lastTime = startTime;

    while (1) {
        time_t currentTime = time(NULL);
        double elapsedTime = difftime(currentTime, lastTime);

        if (elapsedTime >= 5.0) {  // Print every 5 seconds
            double hashesPerSecond = (*hashCount) / elapsedTime;
            printf("Hashes per second: %.2f\n", hashesPerSecond);

            // Reset the counters for the next interval
            *hashCount = 0;
            lastTime = currentTime;
        }
    }

    return NULL; // Required for pthread functions
}

int main() {
    // Charset and maximum length for brute-forcing
    char charset[] = "abcdefghijklmnopqrstuvwxyz0123456789";
    int maxLen = 5; // adjust based on how many characters you want to test

    // Start the stats printing in a separate thread
    pthread_t statsThread;
    pthread_create(&statsThread, NULL, printStats, &hashCount);

    // Start brute-forcing
    bruteForceMD5(charset, maxLen);

    // Wait for the stats thread to finish (it won't, but this is a clean way to join threads)
    pthread_join(statsThread, NULL);

    return 0;
}
