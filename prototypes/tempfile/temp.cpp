#include <system_error>

#include <llvm/Support/Path.h>
#include <llvm/ADT/SmallString.h>
#include <llvm/Support/FileSystem.h>
#include <llvm/Support/raw_ostream.h>

int main()
{
    {
        llvm::SmallString<128> path;
        std::error_code EC = llvm::sys::fs::createUniqueFile( "test-no-path-%%%%%%.tmp", path );
        if ( EC )
        {
            llvm::errs() << "Error: " << EC.message() << "\n";
            return 1;
        }
        llvm::outs() << "Created: " << path << "\n";
    }

    {
        llvm::SmallString<128> td;
        llvm::sys::path::system_temp_directory( true, td );
        llvm::outs() << "System temp dir: " << td << "\n";

        llvm::SmallString<128> p = td;
        llvm::sys::path::append( p, "silly-test-%%%%%%.tmp" );

        llvm::SmallString<128> result;
        auto EC = llvm::sys::fs::createUniqueFile( p, result );
        if ( !EC )
        {
            llvm::outs() << "Created in temp: " << result << "\n";
        }
    }

    return 0;
}
