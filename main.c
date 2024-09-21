#include "md5.h"
#include <stdio.h>
#include <string.h>

void runHashInputLoop();

int main() {
    runHashInputLoop();
    return 0;
}

void runHashInputLoop() {
    char *inputStr;
    char *hash;
    int strLen = 0;
    Blocks *blocks;
    
    // Array of test strings
    const char* testStrings[] = {
        "Hello, World!",
        "Testing MD5 hash",
        "This is a longer string to test",
        "EXIT"
    };
    int numTests = sizeof(testStrings) / sizeof(testStrings[0]);

    for (int testIndex = 0; testIndex < numTests; testIndex++) {
        inputStr = (char*)testStrings[testIndex];
        strLen = strlen(inputStr);

        if (strLen > 1000) {
            printf("Input too long! Limit is 1000 characters.\n\n");
            continue;
        }

        printf("Testing string: %s\n", inputStr);

        blocks = makeBlocks(inputStr, strLen);
        hash = md5(blocks);
        printf("MD5 Hash value: %s\n\n", hash);

        free(hash);

        if (strcmp(inputStr, "EXIT") == 0) {
            break;
        }
    }
}
