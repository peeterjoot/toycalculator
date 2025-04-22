#include <stdio.h>

extern "C"
void __toy_print( double value )
{
    printf( "%f\n", value );
}
