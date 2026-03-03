#!/usr/bin/perl
use strict;
use warnings;

# Usage: generate_failure_test.pl <stem>
#
# Given a stem (e.g. "foo"), reads:
#   foo.compile.out   -- expected compiler error output lines
#   foo.silly.in      -- the source body of the test
#
# Will end up with wrong line numbers, requiring hacking or refresh of .compile.out text and rerun.
#
# Writes foo.silly containing:
#   // RUN: %Not %ExeSilly -g %s -o %t.exe 2>&1 | %FileCheck %s
#   // CHECK: <line1>
#   // CHECK: <line2>
#   ...
#   <contents of foo.silly.in>

die "Usage: $0 <stem>\n" unless @ARGV == 1;

my $stem        = $ARGV[0];
my $expected    = "$stem.compile.out";
my $source_in   = "$stem.silly.in";
my $output      = "$stem.silly";

open( my $exp_fh, '<', $expected )
    or die "Cannot open '$expected': $!\n";

open( my $in_fh, '<', $source_in )
    or die "Cannot open '$source_in': $!\n";

open( my $out_fh, '>', $output )
    or die "Cannot open '$output' for writing: $!\n";

# Emit the RUN line
print $out_fh qq(// RUN: cd %S
// RUN: %Not %ExeSilly -g $stem.silly -o %t.exe 2>&1 | %FileCheck %s
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
