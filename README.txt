
## TODO

1)
./toycalculator ../samples/test.toy
Warning: Variable x not supported at line 4

think this was due to assignment with a variable.

Want semantic checking for the variable to make sure it has a value -- > where did I leave that warning?

2) Unary op is a place holder.  should be +, - (but do it properly in the grammar)

3) LLVM IR lowering.

## Building

### anltlr4 setup (WSL2)

sudo apt update
sudo apt upgrade -y
sudo apt install openjdk-11-jre -y
sudo apt-get install libmlir-19-dev
sudo apt-get install llvm-19-dev
sudo apt-get install mlir-19-tools
sudo apt-get install libantlr4-runtime-dev
sudo apt-get install antlr4

wget https://www.antlr.org/download/antlr-4.10-complete.jar

##wget https://www.antlr.org/download/antlr-4.13.2-complete.jar

### anltlr4 setup (Fedora)

sudo dnf -y install antlr4-runtime antlr4 antlr4-cpp-runtime antlr4-cpp-runtime-devel

### Building MLIR

I needed a custom build of llvm/mlir, as I didn't find a package that had the MLIR tablegen files.  Then had to refine that to enable rtti, as altlr4 uses dynamic_cast<>, so -fno-rtti
breaks it (without -fno-rtti, I was getting typeinfo symbol link errors.)

This is how I build and installed my llvm:

	# git clone https://github.com/llvm/llvm-project.git
	cd ~/llvm-project
	git checkout llvmorg-20.1.3

	BUILDDIR=~/build-llvm
	rm -rf $BUILDDIR
	mkdir $BUILDDIR
	cd $BUILDDIR

	BUILD_TYPE=Debug
	#BUILD_TYPE=Release

	INSTDIR=/usr/local/llvm-20.1.3

	# By default need to compile code that uses llvm/mlir with -fno-rtti, or else we get link errors.  However, antlr4 uses dynamic-cast, so we have to enable rtti if using that.
	TARGETS='X86;ARM'
	cmake \
	-G \
	Ninja \
	../llvm-project/llvm \
	-DBUILD_SHARED_LIBS=true \
	-DCMAKE_BUILD_TYPE=${BUILD_TYPE} \
	-DLLVM_ENABLE_ASSERTIONS=TRUE \
	-DLLVM_OPTIMIZED_TABLEGEN=ON \
	-DLLVM_LIBDIR_SUFFIX=64 \
	-DCMAKE_INSTALL_RPATH=${INSTDIR}/lib64 \
	-DLLVM_TARGETS_TO_BUILD="${TARGETS}" \
	-DCMAKE_INSTALL_PREFIX=${INSTDIR} \
	-DLLVM_ENABLE_RTTI=ON \
	-DLLVM_ENABLE_PROJECTS='mlir'

	ninja

	sudo rm -rf ${INSTDIR}
	sudo ${NINJA} install
