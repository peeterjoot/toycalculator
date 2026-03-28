#include <stdio.h>

int main()
{
    int x = 3; // 5

    if ( x < 4 ) // 7
    {
        int y; // 9
        y = 42; // 10
        printf( "%d\n", y ); // 11
    } // 12:5

    printf( "Done.\n" ); // 14
}
