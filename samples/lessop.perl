my @v1 = (qw(i1 i8 i16 i32 i64 f32 f64));
my @v2 = (qw( j1 j8 j16 j32 j64 g32 g64));
foreach my $v1 (@v1) { foreach my $v2 (@v2) { print "b = $v1 < $v2;\nPRINT b;\n"; } }
