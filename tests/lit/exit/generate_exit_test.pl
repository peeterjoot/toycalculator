#!/usr/bin/perl
use strict;
use warnings;

# Usage: generate_exit_test.pl <stem>
#
# Given a stem (e.g. "foo"), reads:
#   foo.out   -- expected output lines
#   foo.silly.in      -- the source body of the test
#
# Will end up with wrong line numbers, requiring hacking or refresh of .out text and rerun.
#
# Writes foo.silly containing:
#   // RUN: %Not %ExeSilly -g %s -o %t.exe 2>&1 | %FileCheck %s
#   // CHECK: <line1>
#   // CHECK: <line2>
#   ...
#   <contents of foo.silly.in>

die "Usage: $0 <stem>\n" unless @ARGV == 1;

my $stem        = $ARGV[0];
my $expected    = "expected/$stem.out";
my $source_in   = "$stem.silly.in";
my $output      = "$stem.silly";

my %rc = (
'bool' => 1,
'boolr' => 1,
'int32' => 1,
'exit3' => 3,
'exitx' => 3,
'exit42' => 42,
'exitarrayelement' => 42,
);

open( my $exp_fh, '<', $expected )
    or die "Cannot open '$expected': $!\n";

open( my $in_fh, '<', $source_in )
    or die "Cannot open '$source_in': $!\n";

open( my $out_fh, '>', $output )
    or die "Cannot open '$output' for writing: $!\n";

my $rc = $rc{$stem};

# Emit the RUN line
print $out_fh qq(// RUN: %ExeSilly %s -o %t.exe
// RUN: %t.exe || (test \$? -eq $rc || { echo "Expected exit $rc, got \$?"; exit 1; })
);

# Emit one CHECK line per line of expected output
while ( my $line = <$exp_fh> )
{
    chomp $line;
    print $out_fh "// CHECK: $line\n";
}

# Blank separator line
print $out_fh "\n";

# Append the source body
while ( my $line = <$in_fh> )
{
    print $out_fh $line;
}

close $exp_fh;
close $in_fh;
close $out_fh;

print "Written: $output\n";
