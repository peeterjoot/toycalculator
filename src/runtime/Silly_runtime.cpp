///
/// @file    Silly_runtime.cpp
/// @author  Peeter Joot <peeterjoot@pm.me>
/// @brief   Runtime functions for the silly compiler and language.
///
#include <inttypes.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "PrintFlags.hpp"

extern "C"
{
    /// This is the runtime that implements a silly language 'ABORT' statement.
    ///
    /// This prints a FATAL message to stderr and aborts.
    ///
    /// @param len [in] The length of str.
    /// @param file [in] A __FILE__ like variable representing the filename containing the ABORT (not null terminated.)
    /// @param line [in] The line number for the ABORT.
    void __silly_abort( size_t len, const char* file, int line )
    {
        fflush( NULL );
        fprintf( stderr, "%.*s:%d:FATAL ERROR: aborting\n", (int)len, file, line );
        abort();
    }

    /// This is the runtime implementation of a silly language 'PRINT' or 'ERROR' statement.
    void __silly_print( int num_args, const struct silly::PrintArg* args )
    {
        for ( int i = 0; i < num_args; ++i )
        {
            const struct silly::PrintArg& arg = args[i];

            FILE* where = ( arg.flags & silly::PRINT_FLAGS_ERROR ) ? stderr : stdout;
            const char* newline = ( arg.flags & silly::PRINT_FLAGS_CONTINUE ) ? "" : "\n";

            switch ( arg.kind )
            {
                case silly::PrintKind::I64:
                    fprintf( where, "%" PRId64 "%s", arg.i, newline );
                    break;

                case silly::PrintKind::F64:
                    double d;
                    memcpy(&d, &arg.i, sizeof(d));
                    fprintf( where, "%f%s", d, newline );
                    break;

                case silly::PrintKind::STRING:
                    fprintf( where, "%.*s%s", (int)arg.i, arg.ptr, newline );
                    break;

                default:
                    fprintf( stderr, "Unknown PrintKind %u, for argument %d of %d\n", arg.kind, i, num_args );
                    abort();
            }
        }
    }

    /// This is the runtime implementation of a silly GET statement for an INT8 variable.
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

    /// This is the runtime implementation of a silly GET statement for a BOOL variable.
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

    /// This is the runtime implementation of a silly GET statement for an INT16 variable.
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

    /// This is the runtime implementation of a silly GET statement for an INT32 variable.
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

    /// This is the runtime implementation of a silly GET statement for an INT64 variable.
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

    /// This is the runtime implementation of a silly GET statement for an FLOAT32 variable.
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

    /// This is the runtime implementation of a silly GET statement for an FLOAT64 variable.
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
