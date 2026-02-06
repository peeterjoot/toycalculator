// Current versions of clang w/cir don't crash when stdio.h is included
// /usr/local/llvm-21.1.8/bin/clang -emit-cir for.c
#include <stdio.h>
#include <stdlib.h>
#define THRESH (RAND_MAX/2)

int main()
{
    for ( int i = 0 ; i < 10 ; i++ )
    {
        printf("pre: %d\n", i);

        if ( rand() > THRESH )
        {
            printf("continue: %d\n", i);
            continue;
        }
        else if ( rand() < THRESH )
        {
            printf("break: %d\n", i);
            break;
        }

        printf("post: %d\n", i);
    }

    return 0;
}
