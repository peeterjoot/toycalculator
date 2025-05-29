#! /usr/bin/perl

my %v1 = (
   'i1' => 'TRUE',
   'i8' => 10,
   'i16' => 1000,
   'i32' => 100000,
   'i64' => 100000000000,
   'f32' => 1.1,
   'f64' => 0.22,
);
my %v2 = (
   'j1' => 'FALSE',
   'j8' => 1,
   'j16' => 100,
   'j32' => 10000,
   'j64' => 10000000000,
   'g32' => 1.0,
   'g64' => 0.21,
);

open my $toy, ">lessop.toy" or die;
open my $etoy, ">expected/lessop.out" or die;

print $toy qq(//THIS IS A GENERATED TEST CASE (./lessop.perl).  DO NOT EDIT\n
INT32 i;
BOOL b;\n);

my @symbols = sort keys %v1;
push( @symbols, sort keys %v2 );

foreach my $v ( @symbols )
{
    my $type = $v;
    $type =~ s/^[ij]/INT/;
    $type =~ s/^[fg]/FLOAT/;

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
#foreach my $v1 ( (qw(i1)) )
foreach my $v1 ( sort keys %v1 )
{
    #foreach my $v2( (qw(j16)) )
    foreach my $v2( sort keys %v2 )
    {
        my $e, $f;
        my $a = $v1{$v1};
        my $b = $v2{$v2};
        $a =~ s/TRUE/1/;
        $b =~ s/TRUE/1/;
        $a =~ s/FALSE/0/;
        $b =~ s/FALSE/0/;

        if ( $a < $b )
        {
            $e = 1;
        }
        else
        {
            $e = 0;
        }

        if ( $b < $a )
        {
            $f = 1;
        }
        else
        {
            $f = 0;
        }

        my $m = sprintf( "i = %d;\nPRINT i;\n", $i );
        print $etoy "$i\n$e\n";
        print $toy "${m}b = $v1 < $v2;\nPRINT b;\n";
        $i++;

        $m = sprintf( "i = %d;\nPRINT i;\n", $i );
        print $etoy "$i\n$f\n";
        print $toy "${m}b = $v2 < $v1;\nPRINT b;\n";
        $i++;
    }
}

close $toy;
close $etoy;

# vim: et ts=4 sw=4
