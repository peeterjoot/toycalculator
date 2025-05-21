/**
 * @file    prototypes/mlirtest.cpp
 * @author  Peeter Joot <peeterjoot@pm.me>
 * @brief   This is a standalone MLIR -> LLVM-IR generation program, with working DWARF instrumentation, but done wrong.
 *
 * @section Description
 *
 * I wanted a working standalone MLIR -> LLVM-IR file that was standalone, for which line and variable debugging worked
 * end to end. Insertion of the DI info after the fact is wholely and disgustingly wrong.  This shouldn't be required.
 * I want to use this program as the starting point to figure out how to do this the right way -- i.e.: all the location
 * info that is saved in the MLIR layer shouldn't be thrown away (foracbly reconstructed after the fact), so what is the
 * right way to make sure that lowering converts that into proper DIbuilder statements without this hacking?
 *
 */

#include <assert.h>
#include <llvm/IR/DIBuilder.h>
#include <llvm/IR/DebugInfoMetadata.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/Support/CommandLine.h>
#include <llvm/Support/InitLLVM.h>
#include <llvm/Support/raw_ostream.h>
#include <mlir/Conversion/ArithToLLVM/ArithToLLVM.h>
#include <mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h>
#include <mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h>
#include <mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h>
#include <mlir/Target/LLVMIR/Export.h>

#include <mlir/Dialect/DLTI/DLTI.h>
#include <llvm/ADT/TypeSwitch.h>
#include <mlir/Dialect/LLVMIR/Transforms/LegalizeForExport.h>
#include <mlir/Dialect/LLVMIR/Transforms/DIExpressionLegalization.h>
#include <mlir/Target/LLVMIR/ModuleTranslation.h>

#define FAKE_PROGRAM_SOURCE_NAME "mlirtest.proto"

class myDIBuilder
{
   public:
    llvm::DIBuilder diBuilder;

    // The memory for all these pointers is tied to the DIBuilder builder above, and gets freed at the finalize call
    // later:
    llvm::DIFile *file;
    llvm::DICompileUnit *cu;
    llvm::DISubroutineType *subprogramType;
    llvm::DISubprogram *subprogram;
    llvm::DIType *diTypeI64;
    llvm::Function *llvmFunc;

    llvm::BasicBlock &entryBlock;
    llvm::IRBuilder<> llvmBuilder;

    myDIBuilder( std::unique_ptr<llvm::Module> &llvmModule )
        : diBuilder{ *llvmModule },
          file{ diBuilder.createFile( FAKE_PROGRAM_SOURCE_NAME, "" ) },
          cu{ diBuilder.createCompileUnit( llvm::dwarf::DW_LANG_C, file, "MLIR Compiler", false, "", 0 ) },
          subprogramType{ diBuilder.createSubroutineType( diBuilder.getOrCreateTypeArray( {} ) ) },
          subprogram{ diBuilder.createFunction( file, "main", "main", file, 1, subprogramType, 1,
                                                llvm::DINode::FlagZero, llvm::DISubprogram::SPFlagDefinition ) },
          diTypeI64{ diBuilder.createBasicType( "int", 64, llvm::dwarf::DW_ATE_signed ) },
          llvmFunc{ llvmModule->getFunction( "main" ) },
          entryBlock{ llvmFunc->getEntryBlock() },
          llvmBuilder{ &entryBlock }
    {
    }

    void instrumentVariable( const char *varname, unsigned line, unsigned column, llvm::Value *allocaPtr )
    {
        auto *var =
            diBuilder.createAutoVariable( subprogram, varname, file, line, diTypeI64, true /* force emission */ );

        auto *dbgLoc = llvm::DILocation::get( allocaPtr->getContext(), line, 3, subprogram );

        llvmBuilder.SetInsertPoint( &entryBlock, entryBlock.begin() );

        diBuilder.insertDeclare( allocaPtr, var, diBuilder.createExpression(), dbgLoc, llvmBuilder.GetInsertBlock() );
    }
};

// 2:   x = 6;
// 3:   PRINT x;
void buildAssignmentAndPrint( mlir::OpBuilder &builder, mlir::MLIRContext *context, unsigned aline, int value )
{
    auto varLoc = mlir::FileLineColLoc::get( builder.getStringAttr( FAKE_PROGRAM_SOURCE_NAME ), aline, 3 );
    auto printLoc = mlir::FileLineColLoc::get( builder.getStringAttr( FAKE_PROGRAM_SOURCE_NAME ), aline + 1, 3 );

    mlir::Type int64Type = mlir::IntegerType::get( context, 64 );
    mlir::Type pointerType = mlir::LLVM::LLVMPointerType::get( context );

    mlir::Value i64One = builder.create<mlir::LLVM::ConstantOp>( varLoc, int64Type, builder.getI64IntegerAttr( 1 ) );
    mlir::Value constantValue = builder.create<mlir::arith::ConstantOp>( varLoc, builder.getI64IntegerAttr( value ) );
    mlir::Value ptr = builder.create<mlir::LLVM::AllocaOp>( varLoc, pointerType, int64Type, i64One, 4 );
    builder.create<mlir::LLVM::StoreOp>( varLoc, constantValue, ptr );

    mlir::Value val = builder.create<mlir::LLVM::LoadOp>( printLoc, int64Type, ptr );
    builder.create<mlir::func::CallOp>( printLoc, "__toy_print_i64", mlir::TypeRange{}, mlir::ValueRange{ val } );
}

// fork of mlir::translateModuleToLLVMIR

namespace hack
{
  using namespace mlir;
  using namespace mlir::LLVM;

static FailureOr<llvm::DataLayout>
translateDataLayout(DataLayoutSpecInterface attribute,
                    const DataLayout &dataLayout,
                    std::optional<Location> loc = std::nullopt) {
  if (!loc)
    loc = UnknownLoc::get(attribute.getContext());

  // Translate the endianness attribute.
  std::string llvmDataLayout;
  llvm::raw_string_ostream layoutStream(llvmDataLayout);
  for (DataLayoutEntryInterface entry : attribute.getEntries()) {
    auto key = llvm::dyn_cast_if_present<StringAttr>(entry.getKey());
    if (!key)
      continue;
    if (key.getValue() == DLTIDialect::kDataLayoutEndiannessKey) {
      auto value = cast<StringAttr>(entry.getValue());
      bool isLittleEndian =
          value.getValue() == DLTIDialect::kDataLayoutEndiannessLittle;
      layoutStream << "-" << (isLittleEndian ? "e" : "E");
      continue;
    }
    if (key.getValue() == DLTIDialect::kDataLayoutProgramMemorySpaceKey) {
      auto value = cast<IntegerAttr>(entry.getValue());
      uint64_t space = value.getValue().getZExtValue();
      // Skip the default address space.
      if (space == 0)
        continue;
      layoutStream << "-P" << space;
      continue;
    }
    if (key.getValue() == DLTIDialect::kDataLayoutGlobalMemorySpaceKey) {
      auto value = cast<IntegerAttr>(entry.getValue());
      uint64_t space = value.getValue().getZExtValue();
      // Skip the default address space.
      if (space == 0)
        continue;
      layoutStream << "-G" << space;
      continue;
    }
    if (key.getValue() == DLTIDialect::kDataLayoutAllocaMemorySpaceKey) {
      auto value = cast<IntegerAttr>(entry.getValue());
      uint64_t space = value.getValue().getZExtValue();
      // Skip the default address space.
      if (space == 0)
        continue;
      layoutStream << "-A" << space;
      continue;
    }
    if (key.getValue() == DLTIDialect::kDataLayoutStackAlignmentKey) {
      auto value = cast<IntegerAttr>(entry.getValue());
      uint64_t alignment = value.getValue().getZExtValue();
      // Skip the default stack alignment.
      if (alignment == 0)
        continue;
      layoutStream << "-S" << alignment;
      continue;
    }
    emitError(*loc) << "unsupported data layout key " << key;
    return failure();
  }

  // Go through the list of entries to check which types are explicitly
  // specified in entries. Where possible, data layout queries are used instead
  // of directly inspecting the entries.
  for (DataLayoutEntryInterface entry : attribute.getEntries()) {
    auto type = llvm::dyn_cast_if_present<Type>(entry.getKey());
    if (!type)
      continue;
    // Data layout for the index type is irrelevant at this point.
    if (isa<IndexType>(type))
      continue;
    layoutStream << "-";
    LogicalResult result =
        llvm::TypeSwitch<Type, LogicalResult>(type)
            .Case<IntegerType, Float16Type, Float32Type, Float64Type,
                  Float80Type, Float128Type>([&](Type type) -> LogicalResult {
              if (auto intType = dyn_cast<IntegerType>(type)) {
                if (intType.getSignedness() != IntegerType::Signless)
                  return emitError(*loc)
                         << "unsupported data layout for non-signless integer "
                         << intType;
                layoutStream << "i";
              } else {
                layoutStream << "f";
              }
              uint64_t size = dataLayout.getTypeSizeInBits(type);
              uint64_t abi = dataLayout.getTypeABIAlignment(type) * 8u;
              uint64_t preferred =
                  dataLayout.getTypePreferredAlignment(type) * 8u;
              layoutStream << size << ":" << abi;
              if (abi != preferred)
                layoutStream << ":" << preferred;
              return success();
            })
            .Case([&](LLVMPointerType type) {
              layoutStream << "p" << type.getAddressSpace() << ":";
              uint64_t size = dataLayout.getTypeSizeInBits(type);
              uint64_t abi = dataLayout.getTypeABIAlignment(type) * 8u;
              uint64_t preferred =
                  dataLayout.getTypePreferredAlignment(type) * 8u;
              uint64_t index = *dataLayout.getTypeIndexBitwidth(type);
              layoutStream << size << ":" << abi << ":" << preferred << ":"
                           << index;
              return success();
            })
            .Default([loc](Type type) {
              return emitError(*loc)
                     << "unsupported type in data layout: " << type;
            });
    if (failed(result))
      return failure();
  }
  StringRef layoutSpec(llvmDataLayout);
  if (layoutSpec.starts_with("-"))
    layoutSpec = layoutSpec.drop_front();

  return llvm::DataLayout(layoutSpec);
}
static std::unique_ptr<llvm::Module>
prepareLLVMModule(mlir::Operation *m, llvm::LLVMContext &llvmContext,
                  llvm::StringRef name) {

  m->getContext()->getOrLoadDialect<LLVM::LLVMDialect>();

  auto llvmModule = std::make_unique<llvm::Module>(name, llvmContext);
  // ModuleTranslation can currently only construct modules in the old debug
  // info format, so set the flag accordingly.
  llvmModule->setNewDbgInfoFormatFlag(false);
  if (auto dataLayoutAttr =
          m->getDiscardableAttr(LLVM::LLVMDialect::getDataLayoutAttrName())) {
    llvmModule->setDataLayout(cast<StringAttr>(dataLayoutAttr).getValue());
  } else {
    FailureOr<llvm::DataLayout> llvmDataLayout(llvm::DataLayout(""));
    if (auto iface = dyn_cast<DataLayoutOpInterface>(m)) {
      if (DataLayoutSpecInterface spec = iface.getDataLayoutSpec()) {
        llvmDataLayout =
            translateDataLayout(spec, DataLayout(iface), m->getLoc());
      }
    } else if (auto mod = dyn_cast<ModuleOp>(m)) {
      if (DataLayoutSpecInterface spec = mod.getDataLayoutSpec()) {
        llvmDataLayout =
            translateDataLayout(spec, DataLayout(mod), m->getLoc());
      }
    }
    if (failed(llvmDataLayout))
      return nullptr;
    llvmModule->setDataLayout(*llvmDataLayout);
  }
  if (auto targetTripleAttr =
          m->getDiscardableAttr(LLVM::LLVMDialect::getTargetTripleAttrName()))
    llvmModule->setTargetTriple(cast<StringAttr>(targetTripleAttr).getValue());

  return llvmModule;
}

std::unique_ptr<llvm::Module> translateModuleToLLVMIRWithDebug(
    mlir::Operation *module, llvm::LLVMContext &llvmContext,
    llvm::StringRef name = {}, bool disableVerification = false) {

  using namespace mlir;
  using namespace mlir::LLVM;

  if (!satisfiesLLVMModule(module)) {
    module->emitOpError("cannot be translated to an LLVMIR module");
    return nullptr;
  }

  std::unique_ptr<llvm::Module> llvmModule =
      prepareLLVMModule(module, llvmContext, name);
  if (!llvmModule)
    return nullptr;

  LLVM::ensureDistinctSuccessors(module);
  LLVM::legalizeDIExpressionsRecursively(module);

  // Setup for debug info: create DIBuilder and attach compile unit.
  llvm::DIBuilder diBuilder(*llvmModule);

  llvm::DIFile *diFile = nullptr;
  llvm::DICompileUnit *compileUnit = nullptr;

  // Try to grab FileLineColLoc from the module or fallback.
  if (auto loc = module->getLoc().dyn_cast<FileLineColLoc>()) {
    auto filename = loc.getFilename().str();
    auto directory = ".";

    diFile = diBuilder.createFile(filename, directory);
    compileUnit = diBuilder.createCompileUnit(
        llvm::dwarf::DW_LANG_C,
        diFile,
        "MLIR-to-LLVM debug info example",
        /*isOptimized=*/false,
        "",
        0,
        llvm::DICompileUnit::DebugEmissionKind::FullDebug);
  }

  ModuleTranslation translator(module, std::move(llvmModule));
  llvm::IRBuilder<> llvmBuilder(llvmContext);

  if (failed(translator.convertOperation(*module, llvmBuilder)))
    return nullptr;
  if (failed(translator.convertComdats()))
    return nullptr;
  if (failed(translator.convertFunctionSignatures()))
    return nullptr;
  if (failed(translator.convertGlobals()))
    return nullptr;
  if (failed(translator.createTBAAMetadata()))
    return nullptr;
  if (failed(translator.createIdentMetadata()))
    return nullptr;
  if (failed(translator.createCommandlineMetadata()))
    return nullptr;

  // Inject DISubprogram for each LLVM function from MLIR with location.
  for (Operation &op : getModuleBody(module).getOperations()) {
    if (auto funcOp = dyn_cast<LLVM::LLVMFuncOp>(op)) {
      llvm::Function *llvmFunc =
          translator.lookupFunction(funcOp.getName());
      if (!llvmFunc || !funcOp.getLoc().isa<FileLineColLoc>())
        continue;

      FileLineColLoc loc = funcOp.getLoc().cast<FileLineColLoc>();
      std::string name = funcOp.getSymName().str();
      llvm::DISubprogram *subprogram = diBuilder.createFunction(
          diFile,
          name,
          name,
          diFile,
          loc.getLine(),
          diBuilder.createSubroutineType(
              diBuilder.getOrCreateTypeArray({})),
          loc.getLine(),
          llvm::DINode::DIFlags::FlagPrototyped,
          llvm::DISubprogram::SPFlagDefinition);

      llvmFunc->setSubprogram(subprogram);
    }
  }

  for (Operation &o : getModuleBody(module).getOperations()) {
    if (!isa<LLVM::LLVMFuncOp, LLVM::GlobalOp, LLVM::GlobalCtorsOp,
             LLVM::GlobalDtorsOp, LLVM::ComdatOp>(&o) &&
        !o.hasTrait<OpTrait::IsTerminator>() &&
        failed(translator.convertOperation(o, llvmBuilder))) {
      return nullptr;
    }
  }

  if (failed(translator.convertFunctions()))
    return nullptr;

  // Finalize debug info
  if (compileUnit)
    diBuilder.finalize();

  translator.llvmModule->setIsNewDbgInfoFormat(UseNewDbgInfoFormat);

  if (!disableVerification &&
      llvm::verifyModule(*translator.llvmModule, &llvm::errs()))
    return nullptr;

  return std::move(translator.llvmModule);
}
}

int main( int argc, char **argv )
{
    llvm::InitLLVM init( argc, argv );
    llvm::cl::ParseCommandLineOptions( argc, argv, "Standalone MLIR to LLVM-IR code (builder+lowering)\n" );

    // Initialize MLIR context
    mlir::MLIRContext context;
    context.getOrLoadDialect<mlir::func::FuncDialect>();
    context.getOrLoadDialect<mlir::arith::ArithDialect>();
    context.getOrLoadDialect<mlir::LLVM::LLVMDialect>();
    context.getOrLoadDialect<mlir::memref::MemRefDialect>();

    // Register dialect translations
    mlir::registerBuiltinDialectTranslation( context );
    mlir::registerLLVMDialectTranslation( context );

    // Create module
    mlir::OpBuilder builder( &context );
    auto loc = mlir::FileLineColLoc::get( builder.getStringAttr( FAKE_PROGRAM_SOURCE_NAME ), 1, 1 );
    auto module = mlir::ModuleOp::create( loc );

    auto printType = builder.getFunctionType( builder.getI64Type(), {} );
    mlir::func::FuncOp print = builder.create<mlir::func::FuncOp>( loc, "__toy_print_i64", printType,
                                                                   llvm::ArrayRef<mlir::NamedAttribute>{} );    //
    print.setPrivate();    // External linkage
    module.push_back( print );

    auto funcType = builder.getFunctionType( {}, builder.getI32Type() );
    auto func = builder.create<mlir::func::FuncOp>( loc, "main", funcType );
    auto &block = *func.addEntryBlock();
    builder.setInsertionPointToStart( &block );

    // 2:   x = 6;
    // 3:   PRINT x;
    buildAssignmentAndPrint( builder, &context, 2, 6 );

    // 4:   y = 7;
    // 5:   PRINT y;
    buildAssignmentAndPrint( builder, &context, 4, 7 );

    // 6:   z = 42;
    // 7:   PRINT z;
    buildAssignmentAndPrint( builder, &context, 6, 42 );

    // 8:   RETURN;
    auto retLocR = mlir::FileLineColLoc::get( builder.getStringAttr( FAKE_PROGRAM_SOURCE_NAME ), 8, 3 );
    auto zero32 =
        builder.create<mlir::arith::ConstantOp>( retLocR, builder.getI32Type(), builder.getI32IntegerAttr( 0 ) );
    builder.create<mlir::func::ReturnOp>( retLocR, mlir::ValueRange{ zero32 } );

    // Add function to module
    module.push_back( func );

    // Dump module before passes
    mlir::OpPrintingFlags flags;
    flags.printGenericOpForm().enableDebugInfo( true );
    module.print( llvm::outs(), flags );

    // Run passes to lower func.func and arith.constant to LLVMDialect
    mlir::PassManager pm( &context );
    pm.addPass( mlir::createConvertFuncToLLVMPass() );
    pm.addPass( mlir::createArithToLLVMConversionPass() );
    pm.addPass( mlir::createFinalizeMemRefToLLVMConversionPass() );
    if ( failed( pm.run( module ) ) )
    {
        llvm::errs() << "Failed to run conversion passes\n";
        return 1;
    }

    // Dump module after passes
    module.print( llvm::outs(), flags );

    // Translate to LLVM IR
    llvm::LLVMContext llvmContext;
    auto llvmModule = mlir::translateModuleToLLVMIR( module, llvmContext, "example" );
    if ( !llvmModule )
    {
        llvm::errs() << "Failed to translate to LLVM IR\n";
        return 1;
    }

    // Add DWARF debug info flags
    // llvmModule->addModuleFlag( llvm::Module::Warning, "Debug Info Version", llvm::DEBUG_METADATA_VERSION );
    llvmModule->addModuleFlag( llvm::Module::Warning, "Dwarf Version", 5 );    // Match Clang's DWARF 5

    // Create debug metadata with DIBuilder
    myDIBuilder dbi( llvmModule );
    auto mainFunc = llvmModule->getFunction( "main" );
    mainFunc->setSubprogram( dbi.subprogram );

#if 0
    // Add debug locations to instructions
    int assignIndex = 0;
    int printIndex = 0;
    for ( auto &block : *mainFunc )
    {
        for ( auto &inst : block )
        {
            if ( ( inst.getOpcode() == llvm::Instruction::Call ) || ( inst.getOpcode() == llvm::Instruction::Load ) )
            {
                int line = 0;
                switch ( printIndex )
                {
                    case 0:
                        line = 3;    // print(x)
                        break;
                    case 1:
                        line = 5;    // print(y)
                        break;
                    case 2:
                        line = 7;    // print(z)
                        break;
                }
                assert( line );
                inst.setDebugLoc( llvm::DILocation::get( llvmContext, line, 3, dbi.subprogram ) );

                if ( inst.getOpcode() == llvm::Instruction::Call )
                {
                    printIndex++;
                }
            }
            else if ( ( inst.getOpcode() == llvm::Instruction::Alloca ) ||
                      ( inst.getOpcode() == llvm::Instruction::Store ) )
            {
                int line = 0;
                const char *v{};
                switch ( assignIndex )
                {
                    case 0:
                        v = "x";
                        line = 2;    // 2:   x = 6;
                        break;
                    case 1:
                        v = "y";
                        line = 4;    // 4:   y = 7;
                        break;
                    case 2:
                        v = "z";
                        line = 6;    // 6:   z = 42;
                        break;
                }
                assert( v );
                assert( line );
                inst.setDebugLoc( llvm::DILocation::get( llvmContext, line, 3, dbi.subprogram ) );

                if ( inst.getOpcode() == llvm::Instruction::Store )
                {
                    assignIndex++;
                }
                else
                {
                    dbi.instrumentVariable( v, line, 3, &inst );
                }
            }
            else if ( inst.getOpcode() == llvm::Instruction::Ret )
            {
                inst.setDebugLoc( llvm::DILocation::get( llvmContext, 8, 3, dbi.subprogram ) );    // return
            }
            else
            {
                inst.setDebugLoc( llvm::DILocation::get( llvmContext, 1, 1, dbi.subprogram ) );    // default
            }
        }
    }
#endif

    // Finalize debug info
    dbi.diBuilder.finalize();

    // Print LLVM IR
    llvmModule->print( llvm::outs(), nullptr, true );

    // Save LLVM IR to file
    std::error_code EC;
    llvm::raw_fd_ostream os( "output.ll", EC );
    llvmModule->print( os, nullptr );
    os.close();

    return 0;
}

// vim: et ts=4 sw=4
