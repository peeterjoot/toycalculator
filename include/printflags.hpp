#pragma once

#include <stdint.h>

namespace silly
{
// Shared between lowering and runtime:  __silly_print_XXX
enum PRINT_FLAGS : uint32_t
{
    PRINT_FLAGS_NONE = 0,
    PRINT_FLAGS_CONTINUE,
    PRINT_FLAGS_ERROR
};

}
