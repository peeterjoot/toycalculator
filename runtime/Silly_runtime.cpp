/**
 * @file    Silly_runtime.cpp
 * @author  Peeter Joot <peeterjoot@pm.me>
 * @brief   Runtime functions for the silly compiler and language.
 */
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

extern "C"
{
    void __silly_print_string( size_t len, const char* str )
    {
        printf( "%.*s\n", (int)len, str );
    }

    void __silly_print_f64( double value )
    {
        printf( "%f\n", value );
    }


    void __silly_print_i64( int64_t value )
    {
        printf( "%" PRId64 "\n", value );
    }

#if 0
    void __silly_print_f32( float value )
    {
        printf( "%f\n", value );
    }

    void __silly_print_i32( int32_t value )
    {
        printf( "%d\n", value );
    }

    void __silly_print_i16( int16_t value )
    {
        printf( "%hd\n", value );
    }

    void __silly_print_i8( int8_t value )
    {
        printf( "%d\n", (int)value );
    }
#endif

    int8_t __silly_get_i8( void )
    {
        int8_t v;
        int n = scanf( "%" SCNi8, &v );
        if ( n != 1 )
        {
            perror( "scanf failed for int8_t" );
            fprintf( stderr, "Expected an signed 8-bit integer\n" );
            abort();
        }
        return v;
    }

    int8_t __silly_get_i1( void )
    {
        int8_t v = __silly_get_i8();

        if ( ( v != 1 ) && ( v != 0 ) )
        {
            fprintf( stderr, "Fatal runtime error:%s: Boolean has unexpected value: %hhd\n", __func__, v );
            abort();
        }
        return v;
    }

    int16_t __silly_get_i16( void )
    {
        int16_t v;
        int n = scanf( "%" SCNi16, &v );
        if ( n != 1 )
        {
            perror( "scanf failed for int16_t" );
            fprintf( stderr, "Fatal runtime error:%s: Expected an signed 16-bit integer\n", __func__ );
            abort();
        }
        return v;
    }

    int32_t __silly_get_i32( void )
    {
        int32_t v;
        int n = scanf( "%" SCNi32, &v );
        if ( n != 1 )
        {
            perror( "scanf failed for int32_t" );
            fprintf( stderr, "Fatal runtime error:%s: Expected an signed 32-bit integer\n", __func__ );
            abort();
        }
        return v;
    }

    int64_t __silly_get_i64( void )
    {
        int64_t v;
        int n = scanf( "%" SCNi64, &v );
        if ( n != 1 )
        {
            perror( "scanf failed for int64_t" );
            fprintf( stderr, "Fatal runtime error:%s: Expected an signed 64-bit integer\n", __func__ );
            abort();
        }
        return v;
    }

    float __silly_get_f32( void )
    {
        float v;
        int n = scanf( "%f", &v );
        if ( n != 1 )
        {
            perror( "scanf failed for float" );
            fprintf( stderr, "Fatal runtime error:%s: Expected a floating-point number\n", __func__ );
            abort();
        }
        return v;
    }

    double __silly_get_f64( void )
    {
        double v;
        int n = scanf( "%lf", &v );
        if ( n != 1 )
        {
            perror( "scanf failed for double" );
            fprintf( stderr, "Fatal runtime error:%s: Expected a double-precision floating-point number\n", __func__ );
            abort();
        }
        return v;
    }
}
