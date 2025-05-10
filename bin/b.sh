#!/usr/bin/perl

#-----------------------------------------------------------------------------
# POD Format Documentation.  Read "perldoc perlpod" for an example.
# When done, check syntax with "podchecker".

=head1 NAME

b.sh - run ninja and filter output.

=head1 SYNOPSIS

b.sh [--help] [-j N] [-k N]

=head1 DESCRIPTION

Options:

=over 4

=item -j N

Restrict default ninja parallelism.  Example: -j 1

=item -k N

How many errors to tolerate.  Default is 3.

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
my $j;
my $k = 3;

($myName = $0) =~ s@.*[/\\]@@ ;

#Getopt::Long::Configure( 'pass_through' ) ;
GetOptions (
   'help'               => sub { pod2usage(-verbose => 2) ; },
   'j=i'		=> \$j,
   'k=i'		=> \$k,
) or pod2usage(-verbose => 0) ;

my $flags = "-v -k $k";
if ( defined $j )
{
   $flags .= " -j $j";
}

# Validate/handle options, and everything else...
system( qq(ninja $flags 2>&1 | tee o | grep error: | tee e) );

# vim: et ts=3 sw=3
