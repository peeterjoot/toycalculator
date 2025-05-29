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

print qq(//THIS IS A GENERATED TEST CASE (./lessop.perl).  DO NOT EDIT\n
BOOL b;\n);

my @symbols = sort keys %v1;
push( @symbols, sort keys %v2 );

foreach my $v ( @symbols )
{
    my $type = $v;
    $type =~ s/^[ij]/INT/;
    $type =~ s/^[fg]/FLOAT/;

    $type =~ s/INT1$/BOOL/;

    print "$type $v;\n";
}

foreach my $v ( sort keys %v1 )
{
    print "$v = $v1{$v};\n";
}

foreach my $v ( sort keys %v2 )
{
    print "$v = $v2{$v};\n";
}

foreach my $v1 ( sort keys %v1 )
{
    foreach my $v2( sort keys %v2 )
    {
        print "b = $v1 < $v2;\nPRINT b;\n";
        print "b = $v2 < $v1;\nPRINT b;\n";

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
            $f = 0;
        }
        else
        {
            $e = 0;
            $f = 1;
        }

        print "//b = $v1 < $v2;\n";
        print "//$e\n";
        print "//b = $v2 < $v1;\n";
        print "//$f\n";
    }
}

# vim: et ts=4 sw=4
