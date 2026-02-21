rm -f foo*
silly -o foo simpleless.silly
./foo

rm -f foo* moo*
silly -c -o foo.o simpleless.silly
file foo.o
silly -o moo foo.o
./moo

rm -f foo foo.o
silly foo.silly 
ls
./foo

rm -f foo foo.o
silly foo.silly  -c
ls
silly foo.o  -o moo
./moo

rm -rf foo foo.o moo out
silly foo.silly  -c --output-directory out
find out

rm -rf foo foo.o moo out
silly foo.silly  --output-directory out --keep-temp
find out

(cd $TMPDIR && rm -rf foo foo.o moo)
silly foo.silly --keep-temp
ls $TMPDIR
