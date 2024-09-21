#include "md5.h"
#include <stdio.h>
#include <string.h>

void runHashInputLoop();

int main() {
    runHashInputLoop();
    return 0;
}

void runHashInputLoop() {
    const char* testStrings[] = {
        "Hello, World!",
        "Testing MD5 hash",
        "This is a longer string to test",
        "EXIT"
    };
    int numTests = sizeof(testStrings) / sizeof(testStrings[0]);

    for (int testIndex = 0; testIndex < numTests; testIndex++) {
        const char* inputStr = testStrings[testIndex];
        int strLen = strlen(inputStr);

        if (strLen > 1000) {
            printf("Input too long! Limit is 1000 characters.\n\n");
            continue;
        }

        printf("Testing string: %s\n", inputStr);

        // Replace with your MD5 calculation implementation
        char* hash = calculateMD5(inputStr, strLen);

        printf("MD5 Hash value: %s\n\n", hash);

        free(hash);

        if (strcmp(inputStr, "EXIT") == 0) {
            break;
        }
    }
}
