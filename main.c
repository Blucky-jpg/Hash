#include "md5.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Function prototypes
void bruteForceMD5(char* targetHash, int maxLen);
void generateCombination(char* currentStr, int currentPos, int maxLen, char* targetHash);

// Character set for brute force
const char charset[] = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";

// Entry point
int main() {
    char targetHash[33]; // The target MD5 hash you're trying to crack
    printf("Enter the target MD5 hash: ");
    scanf("%32s", targetHash);

    int maxLen = 6; // Maximum length of strings to try

    bruteForceMD5(targetHash, maxLen);

    return 0;
}

// Function to perform brute force search for MD5 match
void bruteForceMD5(char* targetHash, int maxLen) {
    char* currentStr = (char*)malloc((maxLen + 1) * sizeof(char));
    if (!currentStr) {
        printf("Memory allocation failed!\n");
        return;
    }
    
    // Start generating combinations from length 1 to maxLen
    for (int len = 1; len <= maxLen; len++) {
        printf("Trying length %d\n", len);
        generateCombination(currentStr, 0, len, targetHash);
    }

    free(currentStr);
}

// Recursive function to generate all possible combinations of given length
void generateCombination(char* currentStr, int currentPos, int maxLen, char* targetHash) {
    if (currentPos == maxLen) {
        currentStr[currentPos] = '\0'; // Null-terminate the string

        // Generate MD5 hash of the current string
        Blocks *blocks = makeBlocks((ubyte*)currentStr, strlen(currentStr));
        char *generatedHash = md5(blocks);

        // Compare generated hash with target hash
        if (strcmp(generatedHash, targetHash) == 0) {
            printf("Match found: %s -> %s\n", currentStr, generatedHash);
            free(generatedHash);
            exit(0); // Exit once a match is found
        }

        free(generatedHash);
        return;
    }

    // Iterate over the character set
    for (int i = 0; i < sizeof(charset) - 1; i++) {
        currentStr[currentPos] = charset[i];
        generateCombination(currentStr, currentPos + 1, maxLen, targetHash);
    }
}
