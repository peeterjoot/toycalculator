#include <stdio.h>

int main() // 3
{ long myLoopVar = 1; // 4
    for ( ; myLoopVar < 2 ; myLoopVar++ ) // 5
    { // 6
        long myScopeVar; // 7
        myScopeVar = 1 + myLoopVar; // 8
        printf("%ld\n", myScopeVar); // 9
    } // 10

    return 0; // 12
} // 13
