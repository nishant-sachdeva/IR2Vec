//===- IR2Vec.cpp - Top-level driver utility --------------------*- C++ -*-===//
//
// Part of the IR2Vec Project, under the Apache License v2.0 with LLVM
// Exceptions. See the LICENSE file for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CollectIR.h"
#include "FlowAware.h"
#include "Symbolic.h"
#include "Vocabulary.h"
#include "version.h"

#include "llvm/Support/CommandLine.h"
#include "utils.h"
#include <stdio.h>
#include <time.h>

#include "llvm/Support/CommandLine.h"
#include <llvm/Analysis/MemoryDependenceAnalysis.h>
#include "llvm/Analysis/MemorySSA.h"
#include <llvm/IR/BasicBlock.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/Instruction.h>
#include <llvm/IR/Module.h>
#include <llvm/Passes/PassBuilder.h>
#include <llvm/Support/raw_ostream.h>

#include "llvm/Passes/PassPlugin.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Support/SourceMgr.h"

#include <llvm/Analysis/AliasAnalysis.h>
#include <llvm/Analysis/BasicAliasAnalysis.h> // For BasicAA

using namespace llvm;
using namespace IR2Vec;

cl::OptionCategory category("IR2Vec Options");

cl::opt<bool> cl_sym("sym", cl::Optional,
                     cl::desc("Generate Symbolic Encodings"), cl::init(false),
                     cl::cat(category));

cl::opt<bool> cl_fa("fa", cl::Optional,
                    cl::desc("Generate Flow-Aware Encodings"), cl::init(false),
                    cl::cat(category));
cl::opt<bool> cl_printTime("printTime", cl::Optional,
                           cl::desc("Prints time taken to generate Encodings"),
                           cl::init(false), cl::cat(category));
cl::opt<bool> cl_collectIR(
    "collectIR", cl::Optional,
    cl::desc("Generate triplets for training seed embedding vocabulary"),
    cl::init(false), cl::cat(category));
cl::opt<std::string> cl_iname(cl::Positional, cl::desc("Input file path"),
                              cl::Required, cl::cat(category));
cl::opt<unsigned> cl_dim("dim", cl::Optional, cl::init(300),
                         cl::desc("Dimension of the embeddings"),
                         cl::cat(category));
cl::opt<bool> cl_memdep("memdep", cl::Optional,
                        cl::desc("Running mem dep analysis on input .ll file"),
                        cl::init(false), cl::cat(category));

cl::opt<bool> cl_memssa("memssa", cl::Optional,
                        cl::desc("Running mem dep analysis on input .ll file"),
                        cl::init(false), cl::cat(category));

cl::opt<std::string> cl_oname("o", cl::Required, cl::desc("Output file path"),
                              cl::cat(category));
// for on demand generation of embeddings taking function name
cl::opt<std::string> cl_funcName("funcName", cl::Optional, cl::init(""),
                                 cl::desc("Function name"), cl::cat(category));

cl::opt<char>
    cl_level("level", cl::Optional, cl::init(0),
             cl::desc("Level of encoding - p = Program; f = Function"),
             cl::cat(category));

cl::opt<int> cl_cls("class", cl::Hidden, cl::Optional, cl::init(-1),
                    cl::desc("Class information"), cl::cat(category));

cl::opt<float> cl_WO("wo", cl::Hidden, cl::Optional, cl::init(1),
                     cl::desc("Weight of Opcode"), cl::cat(category));

cl::opt<float> cl_WA("wa", cl::Hidden, cl::Optional, cl::init(0.2),
                     cl::desc("Weight of arguments"), cl::cat(category));

cl::opt<float> cl_WT("wt", cl::Hidden, cl::Optional, cl::init(0.5),
                     cl::desc("Weight of types"), cl::cat(category));

cl::opt<bool> cl_debug("debug-ir2vec", cl::Optional,
                       cl::desc("Diagnostics for debugging"), cl::init(false),
                       cl::cat(category));

void printVersion(raw_ostream &ostream) {
  ostream << "\033[1;35m"
          << "IR2Vec Version : " << IR2VEC_VERSION << "\033[0m\n";
  cl::PrintVersionMessage();
}

void generateSymEncodingsFunction(std::string funcName) {
  auto M = getLLVMIR();
  auto vocabulary = VocabularyFactory::createVocabulary(DIM)->getVocabulary();

  IR2Vec_Symbolic SYM(*M, vocabulary);
  std::ofstream o;
  o.open(oname, std::ios_base::app);
  if (printTime) {
    clock_t start = clock();
    SYM.generateSymbolicEncodingsForFunction(&o, funcName);
    clock_t end = clock();
    double elapsed = double(end - start) / CLOCKS_PER_SEC;
    printf("Time taken by on-demand generation of symbolic encodings "
           "is: %.6f "
           "seconds.\n",
           elapsed);
  } else {
    SYM.generateSymbolicEncodingsForFunction(&o, funcName);
  }
  o.close();
}

void generateFAEncodingsFunction(std::string funcName) {
  auto M = getLLVMIR();
  auto vocabulary = VocabularyFactory::createVocabulary(DIM)->getVocabulary();

  IR2Vec_FA FA(*M, vocabulary);
  std::ofstream o, missCount, cyclicCount;
  o.open(oname, std::ios_base::app);
  missCount.open("missCount_" + oname, std::ios_base::app);
  cyclicCount.open("cyclicCount_" + oname, std::ios_base::app);
  if (printTime) {
    clock_t start = clock();
    FA.generateFlowAwareEncodingsForFunction(&o, funcName, &missCount,
                                             &cyclicCount);
    clock_t end = clock();
    double elapsed = double(end - start) / CLOCKS_PER_SEC;
    printf("Time taken by on-demand generation of flow-aware encodings "
           "is: %.6f "
           "seconds.\n",
           elapsed);
  } else {
    FA.generateFlowAwareEncodingsForFunction(&o, funcName, &missCount,
                                             &cyclicCount);
  }
  o.close();
}

void generateFAEncodings() {
  auto M = getLLVMIR();
  auto vocabulary = VocabularyFactory::createVocabulary(DIM)->getVocabulary();

  IR2Vec_FA FA(*M, vocabulary);
  std::ofstream o, missCount, cyclicCount;
  o.open(oname, std::ios_base::app);
  missCount.open("missCount_" + oname, std::ios_base::app);
  cyclicCount.open("cyclicCount_" + oname, std::ios_base::app);
  if (printTime) {
    clock_t start = clock();
    FA.generateFlowAwareEncodings(&o, &missCount, &cyclicCount);
    clock_t end = clock();
    double elapsed = double(end - start) / CLOCKS_PER_SEC;
    printf("Time taken by normal generation of flow-aware encodings "
           "is: %.6f "
           "seconds.\n",
           elapsed);
  } else {
    FA.generateFlowAwareEncodings(&o, &missCount, &cyclicCount);
  }
  o.close();
}

void generateSYMEncodings() {
  auto M = getLLVMIR();
  auto vocabulary = VocabularyFactory::createVocabulary(DIM)->getVocabulary();

  IR2Vec_Symbolic SYM(*M, vocabulary);
  std::ofstream o;
  o.open(oname, std::ios_base::app);
  if (printTime) {
    clock_t start = clock();
    SYM.generateSymbolicEncodings(&o);
    clock_t end = clock();
    double elapsed = double(end - start) / CLOCKS_PER_SEC;
    printf("Time taken by normal generation of symbolic encodings is: "
           "%.6f "
           "seconds.\n",
           elapsed);
  } else {
    SYM.generateSymbolicEncodings(&o);
  }
  o.close();
}

void collectIRfunc() {
  auto M = getLLVMIR();
  CollectIR cir(M);
  std::ofstream o;
  o.open(oname, std::ios_base::app);
  cir.generateTriplets(o);
  o.close();
}

void setGlobalVars(int argc, char **argv) {
  cl::ParseCommandLineOptions(argc, argv);

  fa = cl_fa;
  sym = cl_sym;
  collectIR = cl_collectIR;
  iname = cl_iname;
  oname = cl_oname;
  DIM = cl_dim;
  funcName = cl_funcName;
  level = cl_level;
  cls = cl_cls;
  WO = cl_WO;
  WA = cl_WA;
  WT = cl_WT;
  debug = cl_debug;
  printTime = cl_printTime;
  memdep = cl_memdep;
  memssa = cl_memssa;
}

void checkFailureConditions() {
  bool failed = false;

  if (!(sym || fa || collectIR)) {
    errs() << "Either of sym, fa, or collectIR should be specified\n";
    failed = true;
  }

  if (failed)
    exit(1);

  if (sym || fa) {
    if (level != 'p' && level != 'f') {
      errs() << "Invalid level specified: Use either p or f\n";
      failed = true;
    }
  } else {
    // assert collectIR is True. Else
    assert(collectIR == true);

    if (collectIR && level) {
      errs() << "[WARNING] level would not be used in collectIR mode\n";
    }
  }

  if (failed)
    exit(1);
}

void populateReachingDefs(
  llvm::Instruction* inst, 
  llvm::MemoryDependenceResults* MDR, 
  llvm::SmallVector<const llvm::Instruction*, 10>* RD
);

void getInstORCallPopulate(
  llvm::MemDepResult* memdep,
  llvm::MemoryDependenceResults* MDR,
  llvm::SmallVector<const llvm::Instruction*, 10>* RD
) {
    auto depIns = memdep->getInst();

    if (depIns && !IR2Vec::isLoad(depIns)) {
      // std::cout << "\t" << IR2Vec::getInstStr(depIns);
      RD->push_back(depIns);
    } else {
      populateReachingDefs(depIns, MDR, RD);
    }
}

std::string isDefOrClobber(MemDepResult* memdep) {
  std::string isDefOrClobber = "";
  if (memdep->isLocal()){
    isDefOrClobber = (memdep->isDef()) ? " isDef" : " isClobber";
  } else if (memdep->isNonLocal()) {
    isDefOrClobber = " isNonLocal";
  }
  return isDefOrClobber;
}

void populateReachingDefs(
  llvm::Instruction* inst, 
  llvm::MemoryDependenceResults* MDR, 
  llvm::SmallVector<const llvm::Instruction*, 10>* RD
) {
  // std::cout << "\t" << IR2Vec::getInstStr(inst);
  MemDepResult memdep = MDR->getDependency(inst);
  if (memdep.isLocal()){
    // std::cout << "\t> local " << isDefOrClobber(&memdep) << "\t";
    getInstORCallPopulate(&memdep, MDR, RD);
  } else if (memdep.isNonLocal()) {
    // std::cout << "\t> non-local \n\t\t";
    SmallVector<NonLocalDepResult> nonLocalResults;
    MDR->getNonLocalPointerDependency(inst, nonLocalResults);
    for(auto res: nonLocalResults) {
      auto localmemdep = res.getResult();
      // std::cout << "\t" << isDefOrClobber(&localmemdep) << "\t";
      getInstORCallPopulate(&localmemdep, MDR, RD);
      // std::cout << "\n\t\t";
    }
  } else if (memdep.isNonFuncLocal()) {
    // std::cout << "\t> non-func-local \n\t\t";
    CallBase *CB = dyn_cast<CallBase>(inst);
    if (CB) {
      auto nonLocalDepVec = MDR->getNonLocalCallDependency(CB);
      for(auto vecDep: nonLocalDepVec) {
        auto localmemdep = vecDep.getResult();
        // std::cout << "\t" << isDefOrClobber(&localmemdep) << "\t";
        getInstORCallPopulate(&localmemdep, MDR, RD);
        // std::cout << "\n\t\t";
      }
    }
  } else {
    // std::cout << "\t> unknown";
    assert(memdep.isUnknown() && "Unknown memdep result");
  }

  // std::cout << "\n";

  return;
}

void checkMemdepFunctions(llvm::Module &M) {
  PassBuilder PB;
  FunctionAnalysisManager FAM;

  // We need to initialize the other pass managers even if we don't directly use
  // them
  LoopAnalysisManager LAM;
  CGSCCAnalysisManager CGAM;
  ModuleAnalysisManager MAM;

  // Register all the passes with the PassBuilder
  PB.registerModuleAnalyses(MAM);
  PB.registerCGSCCAnalyses(CGAM);
  PB.registerLoopAnalyses(LAM);
  PB.registerFunctionAnalyses(FAM);

  PB.crossRegisterProxies(LAM, FAM, CGAM, MAM);

  // Register required alias analyses and memory dependence analysis
  FAM.registerPass([] { return MemoryDependenceAnalysis(); });
  FAM.registerPass([] { return BasicAA(); }); // Basic Alias Analysis

  for (auto &F : M) {
    if (!F.isDeclaration()) {
      // std::cout << "ENTERING FOR MEMDEPRESULTS" << std::endl;
      llvm::MemoryDependenceResults &MDR = FAM.getResult<llvm::MemoryDependenceAnalysis>(F);

      // std::cout << "TESTING FOR MEMDEPRESULTS :: MDR ready" << std::endl;
      // std::cout << "getDefaultBlockScanLimit() "  <<
      // MDR.getDefaultBlockScanLimit() << std::endl;

      for (BasicBlock &BB : F) {
        // std::cout << "TESTING FOR MEMDEPRESULTS :: BASIC BLOCK" << std::endl;
        for (Instruction &inst : BB) {
          llvm::SmallVector<const llvm::Instruction*, 10> RD;
          // std::cout << "Checking for Instruction: " << IR2Vec::getInstStr(&inst) << "\n";
          for (unsigned i = 0; i < inst.getNumOperands(); ++i) {
            llvm::Value *operand = inst.getOperand(i);
            
            // Check if the operand is not a pointer
            if (!operand->getType()->isPointerTy()) {
              if(auto parent = dyn_cast<Instruction>(inst.getOperand(i))) {
                RD.push_back(parent);
              }
            }
          }

          populateReachingDefs(&inst, &MDR, &RD);
          if(RD.size() > 0) {
            printReachingDefs(&inst, RD);
          }
        }
      }
    }
  }
}

void printToStdout(const llvm::MemoryAccess *memAccess) {
    std::string output;
    llvm::raw_string_ostream rso(output);  // Create an LLVM stream that writes to a string.
    memAccess->print(rso);                  // Print to raw_string_ostream.
    rso.flush();                           // Ensure all data is written to the string.
    std::cout << output << std::endl;      // Print the string to std::cout.
}


void checkMemssaFunctions(llvm::Module &M) {

  // std::cout << "MemorySSA: Module loaded successfully " << (M.getName()).data() <<
  // std::endl;

  // std::cout << "Instruction Count " << M.getInstructionCount() << std::endl;

  int count = 0;

  PassBuilder PB;
  FunctionAnalysisManager FAM;

  // We need to initialize the other pass managers even if we don't directly use
  // them
  LoopAnalysisManager LAM;
  CGSCCAnalysisManager CGAM;
  ModuleAnalysisManager MAM;

  // Register all the passes with the PassBuilder
  PB.registerModuleAnalyses(MAM);
  PB.registerCGSCCAnalyses(CGAM);
  PB.registerLoopAnalyses(LAM);
  PB.registerFunctionAnalyses(FAM);

  PB.crossRegisterProxies(LAM, FAM, CGAM, MAM);

  // Register required alias analyses and memory dependence analysis
  FAM.registerPass([] { return MemorySSAAnalysis(); });
  FAM.registerPass([] { return BasicAA(); }); // Basic Alias Analysis

  // Run the pass on each function in the module
  for (Function &F : M) {
      if (!F.isDeclaration()) {
          // FPM.run(F, FAM);
            // Get MemorySSA analysis for the function
          MemorySSA &MSSA = FAM.getResult<MemorySSAAnalysis>(F).getMSSA();
          // MSSA.print(errs());

          // Print the memory dependencies for this function

          for (auto &BB : F) {
            for (Instruction &inst : BB) {
              MemoryUseOrDef* useOrDef = MSSA.getMemoryAccess(&inst);
              if (useOrDef) {
                // Print the memory access
                std::cout << "Studying instruction " << IR2Vec::getInstStr(&inst) << "\n";
                MemoryAccess* access = useOrDef->getDefiningAccess();
                
                for(auto i = access->defs_begin(); i!=access->defs_end(); ++i) {
                  if(*i) {
                    if (auto memdef = llvm::dyn_cast<llvm::MemoryDef>(*i)) {
                      auto ins = memdef->getMemoryInst();
                      if(ins)
                        std::cout << "\t\tMemoryDef: " << IR2Vec::getInstStr(ins) << "\n";
                      else
                        std::cout << "No MemoryInst" << "\n";
                    } else if (auto memuse = llvm::dyn_cast<llvm::MemoryUse>(*i)) {
                      auto ins = memuse->getMemoryInst();
                      if(ins)
                        std::cout << "\t\tMemoryUse: " << IR2Vec::getInstStr(ins) << "\n";
                      else
                        std::cout << "No MemoryInst" << "\n";
                    } else if (auto memphi = llvm::dyn_cast<llvm::MemoryPhi>(*i)) {
                      std::cout << "MemPHi" << "\n";
                      for (unsigned num = 0; num < memphi->getNumIncomingValues(); ++num) {
                        MemoryAccess* memphiaccess = memphi->getIncomingValue(num);
                        if (auto memdef = llvm::dyn_cast<llvm::MemoryDef>(memphiaccess)) {
                          auto ins = memdef->getMemoryInst();
                          if(ins)
                            std::cout << "\t\tMemoryDef: " << IR2Vec::getInstStr(ins) << "\n";
                          else
                            std::cout << "No MemoryInst" << "\n";
                        } else if (auto memuse = llvm::dyn_cast<llvm::MemoryUse>(memphiaccess)) {
                          auto ins = memuse->getMemoryInst();
                          if(ins)
                            std::cout << "\t\tMemoryUse: " << IR2Vec::getInstStr(ins) << "\n";
                          else
                            std::cout << "No MemoryInst" << "\n";
                        } else {
                          std::cout << "Try something else - Unknown Memphi" << "\n";
                        }
                      }
                    } else {
                      std::cout << "Try something else" << "\n";
                    }
                  }
                }
                // printToStdout(access);
              }
            }
          }
      }
  }


  // for (auto &F : M) {
  //   count += 1;
  //   if (!F.isDeclaration()) {
  //     // std::cout << "ENTERING FOR MEMDEPRESULTS" << std::endl;
  //     // MemorySSA &MSSA = FAM.getResult<MemorySSAAnalysis>(F);
  //     // std::cout << "Analyzing function: " << F.getName() << "\n";

  //     // std::cout << "TESTING FOR MEMDEPRESULTS :: MDR ready" << std::endl;
  //     // std::cout << "getDefaultBlockScanLimit() "  <<
  //     // MDR.getDefaultBlockScanLimit() << std::endl;

  //     for (BasicBlock &BB : F) {
  //       // std::cout << "TESTING FOR MEMDEPRESULTS :: BASIC BLOCK" << std::endl;
  //       for (Instruction &I : BB) {
  //         MemoryAccess *MA = MSSA.getMemoryAccess(&I);

  //         // if (MA) {
  //         //   std::cout << "Instruction: " << I << "\n";
  //         //   std::cout << "MemoryAccess: " << *MA << "\n";

  //         //   // If the MemoryAccess is a MemoryUseOrDef, get its defining access
  //         //   if (MemoryUseOrDef *MUOD = dyn_cast<MemoryUseOrDef>(MA)) {
  //         //     MemoryAccess *DefiningAccess = MUOD->getDefiningAccess();
  //         //     std::cout << "DefiningAccess: " << *DefiningAccess << "\n";
  //         //   }
  //         // }
  //       }
  //     }
  //   }
  // }
  // std::cout << "Total functions: " << count << std::endl;
}


void runMDA() {
  auto M = getLLVMIR();

  // check if M is a vaid module or not
  if (!M) {
    std::cout << "Invalid module" << std::endl;
    return;
  }

  if (memdep) checkMemdepFunctions(*M);
  else if(memssa) checkMemssaFunctions(*M);

  return;
}

int main(int argc, char **argv) {
  cl::SetVersionPrinter(printVersion);
  cl::HideUnrelatedOptions(category);

  setGlobalVars(argc, argv);

  checkFailureConditions();

  // return 0;

  if (memdep || memssa) {
    runMDA();
    return 0;
  }
  // runMDA();
  // return 0;

  // generateLLVMIR(iname.c_str());

  // std::cout << "Code reached beyond llvm ir output" << std::endl;

  // auto module = Act->getModule();

  // if (module == NULL) {
  //   std::cout << "Error in getModule" << std::endl;
  //   return 0;
  // }

  // // newly added
  if (sym && !(funcName.empty())) {
    generateSymEncodingsFunction(funcName);
  } else if (fa && !(funcName.empty())) {
    generateFAEncodingsFunction(funcName);
  } else if (fa) {
    generateFAEncodings();
  } else if (sym) {
    generateSYMEncodings();
  } else if (collectIR) {
    collectIRfunc();
  }
  // return 0;
}
