#include "md5.h"
#include <getopt.h>
#include <sys/time.h>

void runHashInputLoop();
void runCrackUtility();
bool bruteForcePermutations(int length, int index, char *buffer, char *refHash);

// command line options
// ref: https://www.gnu.org/software/libc/manual/html_node/Getopt-Long-Options.html
// ref: https://www.gnu.org/software/libc/manual/html_node/Getopt-Long-Option-Example.html
// (I also used other similar getopt manuals on gnu.org)
static int helpFlag;
static int testFlag;
static int hashFlag;
static int crackFlag;

static struct option long_options[] =
{
    // all three of these options just set a flag
    {"test", no_argument, &testFlag, 1},
    {"hash", no_argument, &hashFlag, 1},
    {"crack", no_argument, &crackFlag, 1},
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

    // command line argument precedence and restrictions
    // restriction: only one of 'hash' and 'crack' can be used
    // precedence: hash, crack
    if (hashFlag) {
        crackFlag = 0;
    }

    // default if no options are used: run hash input loop
    if (!(helpFlag || testFlag || hashFlag || crackFlag)) {
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
    if (crackFlag) {
        runCrackUtility(5);
    }

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

void runCrackUtility(int maxLength) {
    char refHash[33];
    char buffer[maxLength + 1];
    struct timeval stop, start;
    long timeTaken;

    // get reference hash from user
    puts("Expected hash input format: 32 lowercase hex characters, E.g.: 5d41402abc4b2a76b9719d911017c592");
    puts("Expected plaintext alphabet: [a-z]*");
    printf("Trying up to plaintext length: %d\n\n", maxLength);
    printf("Enter a reference MD5 hash to crack: ");
    fgets(refHash, 33, stdin);
    puts("\nCracking...\n");

    // try all permutations for all lengths of string, up to maxLength
    for (int len = 0; len <= maxLength; len++) {
        printf("Trying all permutations of length %d...\n", len);
        // checking system time in C: https://stackoverflow.com/a/10192994
        gettimeofday(&start, NULL);
        bool matchFound = bruteForcePermutations(len, len - 1, buffer, refHash);
        gettimeofday(&stop, NULL);

        if (matchFound) {
            printf("\nMatch found!\n  Result: '%s'\n\n", buffer);
            return;
        }
        else {
            // print time taken to exhaust all permutations of length 'len'
            timeTaken = (stop.tv_sec - start.tv_sec) * 1000000 + (stop.tv_usec - start.tv_usec);
            printf("  No match, took %lu microseconds\n\n", timeTaken); 
        }
    }

    // no match found for any string of length 0 to maxLength
    puts("No match found!");
}
