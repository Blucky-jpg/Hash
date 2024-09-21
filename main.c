#include "md5.h"
#include <getopt.h>
#include <sys/time.h>

void runHashInputLoop();
bool bruteForcePermutations(int length, int index, char *buffer, char *refHash);

// command line options
// ref: https://www.gnu.org/software/libc/manual/html_node/Getopt-Long-Options.html
// ref: https://www.gnu.org/software/libc/manual/html_node/Getopt-Long-Option-Example.html
// (I also used other similar getopt manuals on gnu.org)
static int testFlag;
static int hashFlag;

static struct option long_options[] =
{
    // all three of these options just set a flag
    {"test", no_argument, &testFlag, 1},
    {"hash", no_argument, &hashFlag, 1},
    // "terminate the array with an element containing all zeros"
    {0, 0, 0, 0}
};

int main(int argc, char **argv) {
    // parse command line arguments
    // (just settings flags, so there's not much to do here)
    int optionIndex;
    int c = 0;

    while (c != -1) {
        c = getopt_long_only(argc, argv, "", long_options, &optionIndex);
    }

    // default if no options are used: run hash input loop
    if (!(testFlag || hashFlag )) {
        hashFlag = 1;
    }

    // run test suite if the 'test' flag option given
    if (testFlag) {
        runTestSuite();
    }

    // run hash input loop if the 'hash' option was given
    if (hashFlag) {
        runHashInputLoop();
    }

    // run crack utility if the 'crack' option was given

    puts("Exiting...\n");

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

bool bruteForcePermutations(int length, int index, char *buffer, char *refHash) {
    if (index < 0) {
        // finished generating string; hash string and compare
        Blocks *blocks = makeBlocks(buffer, length);
        char *hash = md5(blocks);

        // check for a match
        bool match = isHashEqual(hash, refHash);
        if (match) {
            // append null terminator to result
            buffer[length] = '\0';
        }
        // propagate match result
        return match;
    }

    // try all permutations of character at the next index
    for (char c = 'a'; c <= 'z'; c++) {
        buffer[index] = c;
        bool match = bruteForcePermutations(length, index - 1, buffer, refHash);

        if (match) {
            // propagate match found
            return true;
        }
    }

    // tried all permutations at this index without a match
    return false;
}
