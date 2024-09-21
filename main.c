#include "md5.h"
#include <getopt.h>
#include <sys/time.h>

bool bruteForcePermutations(int length, int index, char *buffer, char *refHash);

// command line options
// ref: https://www.gnu.org/software/libc/manual/html_node/Getopt-Long-Options.html
// ref: https://www.gnu.org/software/libc/manual/html_node/Getopt-Long-Option-Example.html
// (I also used other similar getopt manuals on gnu.org)
static int testFlag;
static int hashFlag;

static struct option long_options[] =
{
    {"hash", no_argument, &hashFlag, 1},
    // "terminate the array with an element containing all zeros"
    {0, 0, 0, 0}
};

int main(int argc, char **argv) {
    // (just settings flags, so there's not much to do here)
    int optionIndex;
    int c = 0;

    while (c != -1) {
        c = getopt_long_only(argc, argv, "", long_options, &optionIndex);
    }

    // default if no options are used: run hash input loop
    if (!(hashFlag)) {
        hashFlag = 1;
    }

    // run test suite if the 'test' flag option given
    if (testFlag) {
        runTestSuite();
    }

    puts("Exiting...\n");

    return 0;
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
