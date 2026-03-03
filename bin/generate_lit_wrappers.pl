#!/usr/bin/perl

#-----------------------------------------------------------------------------
# POD Format Documentation.  Read "perldoc perlpod" for an example.
# When done, check syntax with "podchecker".

=head1 NAME

generate_lit_wrappers.pl - <one-line-description>

=head1 SYNOPSIS

generate_lit_wrappers.pl [--help] [<options>]

=head1 DESCRIPTION

Walks through Samples/ and creates lit wrapper tests in tests/ with the same relative subdir structure.

For each foo.silly:
  - creates tests/<subdir>/foo.silly (wrapper)
  - looks for expected/foo.out   → always required
  - looks for expected/foo.stderr.out → optional

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
use strict;
use warnings;
use File::Basename qw(basename dirname);
use File::Path qw(make_path);
use File::Spec;
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

#Getopt::Long::Configure( 'pass_through' ) ;
GetOptions (
    'help'               => sub { pod2usage(-verbose => 2) ; },
) or pod2usage(-verbose => 0) ;

# Validate/handle options, and everything else...

my $SAMPLES_ROOT   = "Samples";
my $LIT_ROOT       = "tests";
my $EXPECTED_SUB   = "expected";                   # where .out files live

my %flags = (
    'arrayprod' => '--init-fill 255',
    'initlist'  => '--init-fill 255',
    'initarray' => '--init-fill 255',
);

my %skip = map { $_ => 1 } (qw(
mod-float-all-types
div-zero-int
));

die "Samples root not found: $SAMPLES_ROOT\n" unless -d $SAMPLES_ROOT;

my @silly_files = glob("$SAMPLES_ROOT/**/*.silly");

for my $src (@silly_files) {
    # Relative path from SAMPLES_ROOT, e.g. "testdir/foo.silly"
    my $rel = File::Spec->abs2rel($src, $SAMPLES_ROOT);

    # e.g. "testdir/foo.silly" → dir = "testdir", base = "foo.silly"
    my $dir  = dirname($rel);
    my $base = basename($rel);           # foo.silly
    my ($stem) = $base =~ /^(.*?)\.silly$/;

    if ( defined $skip{$stem} )
    {
        print "skipping: $rel\n";
        next;
    }

    # Output wrapper path
    my $lit_dir  = File::Spec->catdir($LIT_ROOT, $dir);
    my $lit_file = File::Spec->catfile($lit_dir, "$stem.silly");

    # Expected files
    my $exp_dir = File::Spec->catdir(dirname($src), $EXPECTED_SUB);
    my $exp_out = File::Spec->catfile($exp_dir, "$stem.out");
    my $exp_err = File::Spec->catfile($exp_dir, "$stem.stderr.out");

    make_path($lit_dir) unless -d $lit_dir;

    open my $fh, '>', $lit_file or die "Cannot write $lit_file: $!";

    print $fh qq(// Auto-generated lit wrapper for $SAMPLES_ROOT/$rel
// Regenerate with: generate-lit-wrappers.pl
// Do not edit manually

);

    my $flags = '';
    if ( defined $flags{$stem} )
    {
        $flags = $flags{$stem};
    }
    print $fh "// RUN: \%ExeSilly -g ${flags} \%Examples/$rel -o \%t.exe\n";

    if (-f "$exp_out")
    {
        #print $fh "// RUN: sed 's/^/CHECK: /' < \%Examples/$dir/$EXPECTED_SUB/$stem.out > \%t.stdout.expected\n";

        # stderr is optional
        if (-f "$exp_err") {
            print $fh #"// RUN: sed 's/^/CHECK: /' < \%Examples/$dir/$EXPECTED_SUB/$stem.stderr.out > \%t.stderr.expected\n" .
                      "// RUN: \%t.exe > \%t.out 2> \%t.err\n" .
                      "// RUN: diff -up \%Examples/$dir/$EXPECTED_SUB/$stem.out \%t.out\n" .
                      "// RUN: diff -up \%Examples/$dir/$EXPECTED_SUB/$stem.stderr.out \%t.err\n" .
                      #"// RUN: \%FileCheck \%t.stdout.expected < \%t.out\n" .
                      #"// RUN: \%FileCheck \%t.stderr.expected < \%t.err\n"
                      "";
        } else {
            # only stdout
            print $fh #"// RUN: \%t.exe | \%FileCheck \%t.stdout.expected\n";
                      "// RUN: \%t.exe > \%t.out 2> \%t.err\n" .
                      "// RUN: diff -up \%Examples/$dir/$EXPECTED_SUB/$stem.out \%t.out\n" .
                      "";
        }
    }
    else
    {
        die "NYI: expected file '$exp_err' was found with empty stdout" if (-f "$exp_err");

        print $fh qq(// RUN: \%t.exe > \%t.out 2> \%t.err
// RUN: \%FileCheck \%s --allow-empty --check-prefix=EMPTY < \%t.out

// EMPTY-NOT: ."
);
    }

    close $fh;

    print "Generated: $lit_file\n";
    #last;
}

print "\nDone. Run \$HOME/build-llvm/bin/llvm-lit -v tests/ to execute.\n";

# vim: et ts=4 sw=4
