#include <stdio.h>

int main() {
    int   x = 3; // 4

    if ( x > 4 ) // 6
    {
       printf( "%d\n", x ); // 8
    } // 9
    else if ( x < 5 ) // 10
    {
       printf( "Bug if we don't get here.\n" ); // 12
    }  // 13

    printf( "Done.\n" ); // 15

    return 0;
}
