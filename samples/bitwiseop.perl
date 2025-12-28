#!/usr/bin/perl

#-----------------------------------------------------------------------------
# POD Format Documentation.  Read "perldoc perlpod" for an example.
# When done, check syntax with "podchecker".

=head1 NAME

bitwiseop.perl - generate bitwise (binary operator) test cases.

=head1 SYNOPSIS

bitwiseop.perl [--help] [--or|--and|--xor]

=head1 DESCRIPTION

Options:

=over 4

=item --or

Generate a test case for OR

=item --and

Generate a test case for AND

=item --xor

Generate a test case for XOR

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
my $or = 0;
my $and = 0;
my $xor = 0;
my @origARGV = @ARGV;

($myName = $0) =~ s@.*[/\\]@@ ;

#Getopt::Long::Configure( 'pass_through' ) ;
GetOptions (
   'help'               => sub { pod2usage(-verbose => 2) ; },
   'or!'                => \$or,
   'and!'               => \$and,
   'xor!'               => \$xor,
) or pod2usage(-verbose => 0) ;

# Validate/handle options, and everything else...

my $name = '';
my $op = '';

if ( $or )
{
    $name = 'or';
    $op = 'OR';
}
elsif ( $and )
{
    $name = 'and';
    $op = 'AND';
}
elsif ( $xor )
{
    $name = 'xor';
    $op = 'XOR';
}
else
{
    die 'One of --or, --and, --xor required';
}

my %v1 = (
   'i1'  => 'TRUE',
   'i8'  => 10,
   'i16' => 1000,
   'i32' => 100000,
   'i64' => 100000000000,
);
my %v2 = (
   'j1'  => 'FALSE',
   'j8'  => 1,
   'j16' => 100,
   'j32' => 10000,
   'j64' => 10000000000,
);

open my $silly, ">${name}op.silly" or die;
open my $esilly, ">expected/${name}op.out" or die;

print $silly qq(//THIS IS A GENERATED TEST CASE (./$myName @origARGV).  DO NOT EDIT\n
INT64 r;\n);

my @symbols = sort keys %v1;
push( @symbols, sort keys %v2 );

foreach my $v ( @symbols )
{
    my $type = $v;
    $type =~ s/^[ij]/INT/;
    $type =~ s/INT1$/BOOL/;

    print $silly "$type $v;\n";
}

foreach my $v ( sort keys %v1 )
{
    print $silly "$v = $v1{$v};\n";
}

foreach my $v ( sort keys %v2 )
{
    print $silly "$v = $v2{$v};\n";
}

foreach my $v1 ( sort keys %v1 )
{
    foreach my $v2( sort keys %v2 )
    {
        my $e;
        my $a = $v1{$v1};
        my $b = $v2{$v2};
        $a =~ s/TRUE/1/;
        $b =~ s/TRUE/1/;
        $a =~ s/FALSE/0/;
        $b =~ s/FALSE/0/;

        $a = int( $a );
        $b = int( $b );

        if ( $or )
        {
            $e = ( $a | $b );
        }
        elsif ( $and )
        {
            $e = ( $a & $b );
        }
        elsif ( $xor )
        {
            $e = ( $a ^ $b );
        }

        print $esilly "$v1 ${op} $v2\n$e\n";
        print $silly "PRINT \"$v1 ${op} $v2\";\nr = $v1 ${op} $v2;\nPRINT r;\n";
    }
}

close $silly;
close $esilly;

# vim: et ts=4 sw=4
