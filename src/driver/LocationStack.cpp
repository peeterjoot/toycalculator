/// @file LocationStack.cpp
/// @author Peeter Joot <peeterjoot@pm.me>
/// @brief MLIR location fusion helper code.
#include "LocationStack.hpp"

namespace silly
{
    LocationStack::LocationStack( mlir::OpBuilder &b, mlir::Location loc ) : builder{ b }
    {
        locs.push_back( loc );
    }

    void LocationStack::push_back( mlir::Location loc )
    {
        locs.push_back( loc );
    }

    mlir::Location LocationStack::fuseLocations()
    {
        assert( locs.size() );

        if ( locs.size() == 1 )
        {
            return locs.back();
        }

        return builder.getFusedLoc( locs );
    }
}

// vim: et ts=4 sw=4
