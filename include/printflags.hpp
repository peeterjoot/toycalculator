#pragma once

#include <stdint.h>

// Shared between lowering and runtime:  __silly_print_XXX
enum PRINT_FLAGS : uint32_t
{
    PRINT_FLAGS_NONE = 0,
    PRINT_FLAGS_NEWLINE,
    PRINT_FLAGS_ERROR
};
