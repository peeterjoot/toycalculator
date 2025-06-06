void printf( const char * ); // #include <stdio.h> // -emit-cir bombs if stdio.h is included.
int main( int argc, char** argv ) {
    if ( argc == 1 ) {
        printf( "hi\n" );
    } else {
        printf( "bye\n" );
    }
    return 0;
}
