#!/usr/bin/perl

#-----------------------------------------------------------------------------
# POD Format Documentation.  Read "perldoc perlpod" for an example.
# When done, check syntax with "podchecker".

=head1 NAME

testit.sh - <one-line-description>

=head1 SYNOPSIS

testit.sh [--help] --no-fatal [--just tcname] [--optimize]

=head1 DESCRIPTION

Options:

=over 4

=item --no-fatal

Warn, insted of die, on error.

=item --just testname

Run only testname.  Given full source test.toy, say, this is the test file stem test.

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

($myName = $0) =~ s@.*[/\\]@@ ;
my $fatal = 1;
my $just;
my $optimize = 0;

#Getopt::Long::Configure( 'pass_through' ) ;
GetOptions (
   'help'               => sub { pod2usage(-verbose => 2) ; },
   'fatal!'             => \$fatal,
   'just=s'             => \$just,
   'optimize!'          => \$optimize,
) or pod2usage(-verbose => 0) ;

# Validate/handle options, and everything else...

my $flags;

if ( $optimize )
{
    $flags = '-O 2';
}
else
{
    $flags = '-g';
}

system( qq(rm -rf out) );

my @tests = (qw(
converti
addi
types
test
unary
bool
exit3
exitx
empty
simplest
dcl
foo
bin
));

my %expectedRC = (
   'bool' => 1,
   'exit3' => 3,
   'exitx' => 3,
);

my @warnings = ();

my $pwd = `pwd` ; chomp $pwd;
foreach my $stem (@tests)
{
    next if ( defined $just and $just ne $stem );

    print "##########################################################################\n";
    print "// $stem.toy\n";
    system( qq(cat $stem.toy) );

    print "##########################################################################\n";
    print "../build/toycalculator --output-directory out -g $stem.toy $flags --emit-llvm --emit-mlir\n";
    system( qq(../build/toycalculator --output-directory out -g $stem.toy $flags --emit-llvm --emit-mlir) );

    print "objdump -dr out/${stem}.o\n";
    system( qq(objdump -dr out/${stem}.o) );

    print "clang -g -o out/${stem} out/${stem}.o -L ../build -l toy_runtime -Wl,-rpath,${pwd}/../build\n";
    system( qq(clang -g -o out/${stem} out/${stem}.o -L ../build -l toy_runtime -Wl,-rpath,${pwd}/../build) );

    print "./out/${stem}\n";
    system( qq(./out/${stem} > out/${stem}.out) );
    my $rc = $? >> 8;
    system( qq(cat out/${stem}.out) );

    print "${stem}: RC = $rc\n\n\n\n\n";

    my $erc = $expectedRC{$stem};
    if ( !defined $erc )
    {
        $erc = 0;
    }

    complain( "stem: $rc != $erc" ) if ( $rc ne $erc );

    if ( -e "expected/$stem.out" )
    {
        system( qq(cmp -s expected/${stem}.out out/${stem}.out) );
        my $crc = $? >> 8;
        complain( "cmp -s expected/${stem}.out out/${stem}.out: $crc\n") if ( $crc );
    }
    else
    {
        warn "COMPARE FILE NOT FOUND: expected/$stem.out\n";
    }
}

if ( scalar( @warnings ) )
{
    print "ERRORS:\n\n";
}

foreach ( @warnings )
{
    warn $_;
}

exit 0;

sub complain
{
    my ($message) = @_;

    if ( $fatal )
    {
        die $message;
    }
    else
    {
        push( @warnings, $message );
    }
}

# vim: et ts=4 sw=4
