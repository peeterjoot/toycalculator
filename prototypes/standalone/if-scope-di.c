// mkcir if-scope-di.c
// mkll if-scope-di.c
void foo(int, double);

int main( int argc, char ** argv )
{
    if ( argc )
    {
        int x = 1;
        foo(x, 0.0);
    }
    else
    {
        double x = 42.0;
        foo(0, x);
    }

    return 0;
}
