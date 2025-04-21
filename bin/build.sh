#!/usr/bin/perl

#-----------------------------------------------------------------------------
# POD Format Documentation.  Read "perldoc perlpod" for an example.
# When done, check syntax with "podchecker".

=head1 NAME

build.sh - <one-line-description>

=head1 SYNOPSIS

build.sh [--help] [<options>]

=head1 DESCRIPTION


Options:

=over 4

=item --foo=bar

Blah.

=back

=head1 SUPPORTED PLATFORMS

 Unix (Linux verified)

=head1 SUPPORT

 Send email to peeterjoot@pm.me

=head1 AUTHORS

 Peeter Joot

=cut

#-----------------------------------------------------------------------------

use strict ;
use warnings ;
use Getopt::Long ;
use Pod::Usage ;

# Suppress sourcing of users' .bashrc files in invoked shells
delete $ENV{'ENV'} ;
delete $ENV{'BASH_ENV'} ;

# Set STDOUT and STDERR to unbuffered
select STDERR ; $| = 1 ;
select STDOUT ; $| = 1 ;

my $myName = '' ;
my $doscope = 0;
my $dobuild = 1;
($myName = $0) =~ s@.*[/\\]@@ ;

#Getopt::Long::Configure( 'pass_through' ) ;
GetOptions (
   'help'               => sub { pod2usage(-verbose => 2) ; },
   'cscope!'		=> \$doscope,
   'build!'		=> \$dobuild,
) or pod2usage(-verbose => 0) ;

# Didn't pick consistent install prefixes for my mlir builds (WSL2/Ubuntu, and Fedora-42)... find it:
my @prefix = <"/usr/local/llvm-*">;
die unless ( scalar(@prefix) eq 1 );
my $prefix = $prefix[0];

# Validate/handle options, and everything else...
if ( $dobuild ) {
   mysystem(qq(
set -x

rm -rf build
mkdir build
cd build

cmake -G Ninja -DLLVM_DIR=$prefix/lib64/cmake/llvm -DMLIR_DIR=$prefix/lib64/cmake/mlir ..

ninja -v -k 3 -j 1 2>&1 | tee o
));
}

if ( $doscope ) {
   mysystem( "my_cscope --build --searchdir $prefix --verbose" );
}

exit 0;

sub mysystem
{
   my ($cmd) = @_;

   print "# $cmd\n";

   system( $cmd );
}

# vim: et ts=3 sw=3
