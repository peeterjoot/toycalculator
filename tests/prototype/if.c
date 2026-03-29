#include <stdio.h>

int main(int argc, char ** argv)
{
    //int x = argc;
    int x = 3;

    if ( x < 4 )
    //if ( long myIfVar = argc )
    //if ( argc )
    {
        long myScopeVar;
        myScopeVar = 1 + argc;
        printf("%ld\n", myScopeVar);
    }

    return x;
}
