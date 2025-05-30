#!/usr/bin/perl

#-----------------------------------------------------------------------------
# POD Format Documentation.  Read "perldoc perlpod" for an example.
# When done, check syntax with "podchecker".

=head1 NAME

compareop.perl - generate comparison (binary operator) test cases.

=head1 SYNOPSIS

compareop.perl [--help] [--lt|--le|--eq|--ne]

=head1 DESCRIPTION

Options:

=over 4

=item --lt

Generate a test case for '<'

=item --le

Generate a test case for '<='

=item --eq

Generate a test case for '=='

=item --ne

Generate a test case for '!='

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
my $lt = 0;
my $le = 0;
my $eq = 0;
my $ne = 0;
my @origARGV = @ARGV;

($myName = $0) =~ s@.*[/\\]@@ ;

#Getopt::Long::Configure( 'pass_through' ) ;
GetOptions (
   'help'               => sub { pod2usage(-verbose => 2) ; },
   'lt!'                => \$lt,
   'le!'                => \$le,
   'eq!'                => \$eq,
   'ne!'                => \$ne,
) or pod2usage(-verbose => 0) ;

# Validate/handle options, and everything else...

my $name = '';
my $op = '';

if ( $lt )
{
    $name = 'less';
    $op = '<';
}
elsif ( $le )
{
    $name = 'lesseq';
    $op = '<=';
}
elsif ( $eq )
{
    $name = 'eq';
    $op = 'EQ';
}
elsif ( $ne )
{
    $name = 'neq';
    $op = 'NE';
}
else
{
    die 'One of --lt, --le, --eq, or --ne required';
}

my %v1 = (
   'i1'  => 'TRUE',
   'i8'  => 10,
   'i16' => 1000,
   'i32' => 100000,
   'i64' => 100000000000,
   'f32' => 1.1,
   'f64' => 0.22,
   'k8'  => -10,
   'k16' => -1000,
   'k32' => -100000,
   'k64' => -100000000000,
   'h32' => -1.1,
   'h64' => -0.22,
);
my %v2 = (
   'j1'  => 'FALSE',
   'j8'  => 1,
   'j16' => 100,
   'j32' => 10000,
   'j64' => 10000000000,
   'g32' => 1.0,
   'g64' => 0.21,
   'l8'  => -1,
   'l16' => -100,
   'l32' => -10000,
   'l64' => -10000000000,
   'm32' => -1.0,
   'm64' => -0.21,
);

open my $toy, ">${name}op.toy" or die;
open my $etoy, ">expected/${name}op.out" or die;

print $toy qq(//THIS IS A GENERATED TEST CASE (./$myName @origARGV).  DO NOT EDIT\n
INT32 i;
BOOL b;\n);

my @symbols = sort keys %v1;
push( @symbols, sort keys %v2 );

foreach my $v ( @symbols )
{
    my $type = $v;
    $type =~ s/^[ijkl]/INT/;
    $type =~ s/^[fghm]/FLOAT/;

    $type =~ s/INT1$/BOOL/;

    print $toy "$type $v;\n";
}

foreach my $v ( sort keys %v1 )
{
    print $toy "$v = $v1{$v};\n";
}

foreach my $v ( sort keys %v2 )
{
    print $toy "$v = $v2{$v};\n";
}

my $i = 12340000;
foreach my $v1 ( sort keys %v1 )
{
    foreach my $v2( sort keys %v2 )
    {
        my ($e, $f);
        my $a = $v1{$v1};
        my $b = $v2{$v2};
        $a =~ s/TRUE/1/;
        $b =~ s/TRUE/1/;
        $a =~ s/FALSE/0/;
        $b =~ s/FALSE/0/;

        $a = int( $a ) unless ( $a =~ /\./ );
        $b = int( $b ) unless ( $b =~ /\./ );

        if ( $lt )
        {
            $e = ( $a < $b ) ? 1 : 0;
            $f = ( $b < $a ) ? 1 : 0;
        }
        elsif ( $le )
        {
            $e = ( $a <= $b ) ? 1 : 0;
            $f = ( $b <= $a ) ? 1 : 0;
        }
        elsif ( $eq )
        {
            $e = ( $a == $b ) ? 1 : 0;
            $f = ( $b == $a ) ? 1 : 0;
        }
        elsif ( $ne )
        {
            $e = ( $a != $b ) ? 1 : 0;
            $f = ( $b != $a ) ? 1 : 0;
        }

        my $m = sprintf( "i = %d;\nPRINT i;\n", $i );
        print $etoy "$i\n$e\n";
        print $toy "${m}b = $v1 ${op} $v2;\nPRINT b;\n";
        $i++;

        $m = sprintf( "i = %d;\nPRINT i;\n", $i );
        print $etoy "$i\n$f\n";
        print $toy "${m}b = $v2 ${op} $v1;\nPRINT b;\n";
        $i++;
    }
}

close $toy;
close $etoy;

# vim: et ts=4 sw=4
