#include <stdio.h>

int main()
{
    int   x = 3;

    if ( x < 4 )
    {
       printf( "%d\n", x );
    }
    else if ( x > 5 )
    {
       printf( "Bug if we get here.\n" );
    }
    else
    {
       printf( "Bug if we get here too.\n" );
    }

    printf( "%d\n", 42 );

    return 0;
}
