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

#include "utils.h"
#include "llvm/Support/CommandLine.h"
#include <stdio.h>
#include <time.h>

#include "llvm/Analysis/MemorySSA.h"
#include "llvm/Support/CommandLine.h"
#include <llvm/Analysis/MemoryDependenceAnalysis.h>
#include <llvm/IR/BasicBlock.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/Instruction.h>
#include <llvm/IR/Module.h>
#include <llvm/Passes/PassBuilder.h>
#include <llvm/Support/raw_ostream.h>

#include "llvm/Passes/PassPlugin.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Transforms/Scalar.h"

#include <llvm/Analysis/AliasAnalysis.h>
#include <llvm/Analysis/BasicAliasAnalysis.h> // For BasicAA
#include <llvm/Analysis/DependenceAnalysis.h>

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

void populateRDWithMemDep(
    llvm::Instruction *inst, llvm::MemDepResult *memdep,
    llvm::MemoryDependenceResults *MDR, llvm::DependenceInfo &DA,
    llvm::SmallVector<const llvm::Instruction *, 10> *RD,
    std::unordered_map<const llvm::Instruction *, bool> &Visited);

template <typename T> void printObject(const T *obj) {
  std::string output;
  llvm::raw_string_ostream rso(output);
  obj->print(rso); // Call the `print` method of the object
  rso.flush();
  std::cout << output << std::endl;
}

void printOperand(llvm::Value *operand) {
  std::cout << "Operand: ";
  printObject(operand);

  if (auto *inst = dyn_cast<Instruction>(operand)) {
    std::cout << "Instruction: " << IR2Vec::getInstStr(inst);
  } else if (auto *arg = dyn_cast<Argument>(operand)) {
    std::cout << "Argument: " << (arg->getParent()->getName()).data() << " "
              << arg->getArgNo();
  } else if (auto *constInst = dyn_cast<Constant>(operand)) {
    std::cout << "Constant: " << constInst->getValueID();
  } else {
    std::cout << "Unknown operand type";
  }
  std::cout << std::endl;
}

bool isAlloca(llvm::Instruction *inst) {
  std::string name = inst->getOpcodeName();
  return name == "alloca";
}

void collectNonDepRD(llvm::Instruction *inst,
                     llvm::SmallVector<const llvm::Instruction *, 10> *RD) {
  IR2VEC_DEBUG(std::cout << "\tCollecting Non-Load/Store memDep\t\n");
  // IR2VEC_DEBUG(std::cout << "\t\t" << inst->getOpcodeName() << std::endl);
  for (unsigned i = 0; i < inst->getNumOperands(); ++i) {
    llvm::Value *operand = inst->getOperand(i);
    // IR2VEC_DEBUG(printOperand(operand));
    if (auto parent = dyn_cast<Instruction>(operand)) {
      RD->push_back(parent);
    }
  }
}

std::string memdepType(MemDepResult *memdep) {
  std::string memDepType = "";
  if (memdep->isLocal()) {
    memDepType = (memdep->isDef()) ? " isDef" : " isClobber";
  } else if (memdep->isNonLocal()) {
    memDepType = " isNonLocal";
  } else if (memdep->isNonFuncLocal()) {
    memDepType = " isNonFuncLocal";
  } else if (memdep->isUnknown()) {
    memDepType = " Unknown ";
  }
  return memDepType;
}

void addValueOperands(
    llvm::Instruction *inst,
    llvm::SmallVector<const llvm::Instruction *, 10> *RD,
    std::unordered_map<const llvm::Instruction *, bool> &Visited) {
  for (unsigned i = 0; i < inst->getNumOperands(); ++i) {
    llvm::Value *operand = inst->getOperand(i);
    printObject(operand);
    if (!operand->getType()->isPointerTy()) {
      IR2VEC_DEBUG(std::cout << "\t\tOperand is not a pointer" << std::endl);
      if (auto parent = dyn_cast<Instruction>(inst->getOperand(i))) {
        if (Visited.find(parent) == Visited.end()) {
          Visited[parent] = true;
          RD->push_back(parent);
        }
      }
    }
  }
}

void localMDHandler(
    llvm::Instruction *inst, llvm::MemDepResult *memdep,
    llvm::MemoryDependenceResults *MDR, llvm::DependenceInfo &DA,
    llvm::SmallVector<const llvm::Instruction *, 10> *RD,
    std::unordered_map<const llvm::Instruction *, bool> &Visited) {
  assert(memdep->isLocal() && "We should have a local memdep result");
  llvm::Instruction *depIns = memdep->getInst();
  if (!depIns) {
    IR2VEC_DEBUG(std::cout << "\t> local - nullptr - Exiting" << std::endl);
    return;
  }

  if (Visited.find(depIns) != Visited.end()) {
    IR2VEC_DEBUG(std::cout << "\t Already Visited "
                           << IR2Vec::getInstStr(depIns));
    return;
  } else {
    Visited[depIns] = true;
  }

  std::unique_ptr<Dependence> dependence = DA.depends(inst, depIns, true);
  if (isAlloca(depIns) ||
      (dependence && (dependence->isOutput() || dependence->isAnti()))) {
    IR2VEC_DEBUG(std::cout << IR2Vec::getInstStr(depIns)
                           << "\t> local - Output/Anti Dep - Exiting"
                           << std::endl);
    RD->push_back(depIns);
    return;
  } else {
    IR2VEC_DEBUG(
        std::cout << "\t> local - Not Output/Anti Dep - Checking further"
                  << std::endl);
    addValueOperands(depIns, RD, Visited);
    llvm::MemDepResult localDep = MDR->getDependency(depIns);
    populateRDWithMemDep(depIns, &localDep, MDR, DA, RD, Visited);
  }
}

void nonLocalMDHandler(
    llvm::Instruction *inst, llvm::MemDepResult *memdep,
    llvm::MemoryDependenceResults *MDR, llvm::DependenceInfo &DA,
    llvm::SmallVector<const llvm::Instruction *, 10> *RD,
    std::unordered_map<const llvm::Instruction *, bool> &Visited) {
  assert(memdep->isNonLocal() && "We should have a non-local memdep result");
  SmallVector<NonLocalDepResult> nonLocalResults;
  MDR->getNonLocalPointerDependency(inst, nonLocalResults);
  for (NonLocalDepResult res : nonLocalResults) {
    MemDepResult localmemdep = res.getResult();
    IR2VEC_DEBUG(std::cout << "\t" << memdepType(&localmemdep) << "\t");
    populateRDWithMemDep(inst, &localmemdep, MDR, DA, RD, Visited);
    IR2VEC_DEBUG(std::cout << "\n\t\t");
  }
}

void nonLocalCallHandler(
    llvm::Instruction *inst, llvm::MemDepResult *memdep,
    llvm::MemoryDependenceResults *MDR, llvm::DependenceInfo &DA,
    llvm::SmallVector<const llvm::Instruction *, 10> *RD,
    std::unordered_map<const llvm::Instruction *, bool> &Visited) {
  assert(memdep->isNonFuncLocal() &&
         "We should have a non-local memdep result");
  CallBase *CB = dyn_cast<CallBase>(inst);
  if (CB) {
    auto nonLocalDepVec = MDR->getNonLocalCallDependency(CB);
    for (auto vecDep : nonLocalDepVec) {
      auto localmemdep = vecDep.getResult();
      IR2VEC_DEBUG(std::cout << "\t" << memdepType(&localmemdep) << "\t");

      populateRDWithMemDep(inst, &localmemdep, MDR, DA, RD, Visited);
      IR2VEC_DEBUG(std::cout << "\n\t\t");
    }
  } else {
    IR2VEC_DEBUG(
        std::cout << "\t> " << IR2Vec::getInstStr(inst)
                  << " - Not a call instruction - Collecting NonDepRD\n\t\t");
    collectNonDepRD(inst, RD);
  }
}

void populateRDWithMemDep(
    llvm::Instruction *inst, llvm::MemDepResult *memdep,
    llvm::MemoryDependenceResults *MDR, llvm::DependenceInfo &DA,
    llvm::SmallVector<const llvm::Instruction *, 10> *RD,
    std::unordered_map<const llvm::Instruction *, bool> &Visited) {

  if (memdep->isLocal()) {
    IR2VEC_DEBUG(std::cout << "\t> local " << memdepType(memdep) << "\t");
    localMDHandler(inst, memdep, MDR, DA, RD, Visited);
  } else if (memdep->isNonLocal()) {
    IR2VEC_DEBUG(std::cout << "\t> non-local "
                           << "\n\t\t");
    nonLocalMDHandler(inst, memdep, MDR, DA, RD, Visited);

  } else if (memdep->isNonFuncLocal()) {
    IR2VEC_DEBUG(std::cout << "\t> non-func-local \n\t\t");
    nonLocalCallHandler(inst, memdep, MDR, DA, RD, Visited);
  } else {
    IR2VEC_DEBUG(std::cout << "\t> unknown");
    assert(memdep->isUnknown() && "Unknown memdep result");
  }

  IR2VEC_DEBUG(std::cout << "\n");

  return;
}

void calcReachingDefs(llvm::Instruction *inst,
                      llvm::MemoryDependenceResults &MDR,
                      llvm::DependenceInfo &DA,
                      llvm::SmallVector<const llvm::Instruction *, 10> *RD) {
  IR2VEC_DEBUG(std::cout << "\nStudying instruction "
                         << IR2Vec::getInstStr(inst) << "\n");
  if (!isLoadorStore(inst)) {
    collectNonDepRD(inst, RD);
  } else {
    std::unordered_map<const llvm::Instruction *, bool> Visited;
    Visited[inst] = true;

    addValueOperands(inst, RD, Visited);

    IR2VEC_DEBUG(std::cout << "\t" << IR2Vec::getInstStr(inst));
    MemDepResult memdep = MDR.getDependency(inst);
    populateRDWithMemDep(inst, &memdep, &MDR, DA, RD, Visited);
  }
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
  FAM.registerPass([] { return DependenceAnalysis(); });
  FAM.registerPass([] { return BasicAA(); }); // Basic Alias Analysis

  for (auto &F : M) {
    if (!F.isDeclaration()) {
      llvm::MemoryDependenceResults &MDR =
          FAM.getResult<llvm::MemoryDependenceAnalysis>(F);

      llvm::DependenceInfo &DA = FAM.getResult<llvm::DependenceAnalysis>(F);

      for (BasicBlock &BB : F) {
        for (Instruction &inst : BB) {
          llvm::SmallVector<const llvm::Instruction *, 10> RD;
          calcReachingDefs(&inst, MDR, DA, &RD);
          if (RD.size() > 0) {
            printReachingDefs(&inst, RD);
          }
        }
      }
    }
  }
}

void populateRDWithMemssa(
    llvm::MemoryUseOrDef *useOrDef,
    llvm::SmallVector<const llvm::Instruction *, 10> *RD) {
  llvm::Instruction *inst = useOrDef->getMemoryInst();
  MemoryAccess *access = useOrDef->getDefiningAccess();

  for (auto i = access->defs_begin(); i != access->defs_end(); ++i) {
    if (*i) {
      if (auto memdef = llvm::dyn_cast<llvm::MemoryDef>(*i)) {
        IR2VEC_DEBUG(std::cout << "\t\tMemoryDef:\t");
        // if (memdef) printObject(memdef); else {
        //   IR2VEC_DEBUG(std::cout << "No memdef access" << "\n");
        // };
        auto ins = memdef->getMemoryInst();
        // IR2VEC_DEBUG(std::cout << IR2Vec::getInstStr(ins) << "\n");
        // MemoryAccess* defAccess = memdef->getDefiningAccess();
        // if(defAccess) printObject(defAccess); else {
        //   IR2VEC_DEBUG(std::cout << "No def access" << "\n");
        // }
        if (ins) {
          RD->push_back(ins);
          IR2VEC_DEBUG(std::cout << IR2Vec::getInstStr(ins) << "\n");
        } else {
          IR2VEC_DEBUG(std::cout << "No def inst"
                                 << "\n");
        }
      } else if (auto memuse = llvm::dyn_cast<llvm::MemoryUse>(*i)) {
        IR2VEC_DEBUG(std::cout << "\t\tMemoryUse:\t");
        auto ins = memuse->getMemoryInst();
        if (ins) {
          RD->push_back(ins);
          IR2VEC_DEBUG(std::cout << IR2Vec::getInstStr(ins) << "\n");
        } else {
          IR2VEC_DEBUG(std::cout << "No use inst"
                                 << "\n");
        }
      } else if (auto memphi = llvm::dyn_cast<llvm::MemoryPhi>(*i)) {
        IR2VEC_DEBUG(std::cout << "MemPHi"
                               << "\n");
        for (unsigned num = 0; num < memphi->getNumIncomingValues(); ++num) {
          MemoryAccess *memphiaccess = memphi->getIncomingValue(num);
          if (auto memdef = llvm::dyn_cast<llvm::MemoryDef>(memphiaccess)) {
            IR2VEC_DEBUG(std::cout << "\t\tMemoryDef:\t");
            auto ins = memdef->getMemoryInst();
            if (ins) {
              RD->push_back(ins);
              IR2VEC_DEBUG(std::cout << IR2Vec::getInstStr(ins) << "\n");
            } else
              IR2VEC_DEBUG(std::cout << "No def inst - inside memphi\n");
          } else if (auto memuse =
                         llvm::dyn_cast<llvm::MemoryUse>(memphiaccess)) {
            IR2VEC_DEBUG(std::cout << "\t\tMemoryUse:\t");
            auto ins = memuse->getMemoryInst();
            if (ins) {
              RD->push_back(ins);
              IR2VEC_DEBUG(std::cout << IR2Vec::getInstStr(ins) << "\n");
            } else
              IR2VEC_DEBUG(std::cout << "No Use inst inside memphi"
                                     << "\n");
          } else {
            IR2VEC_DEBUG(std::cout << "Try something else - Unknown Memphi"
                                   << "\n");
          }
        }
      } else {
        IR2VEC_DEBUG(std::cout << "Try something else"
                               << "\n");
      }
    }
  }
}

void calcSSAReachingDefs(llvm::Instruction *inst, llvm::MemorySSA &MSSA,
                         llvm::SmallVector<const llvm::Instruction *, 10> *RD) {
  IR2VEC_DEBUG(std::cout << "Studying instruction " << IR2Vec::getInstStr(inst)
                         << "\n");
  if (!isLoadorStore(inst)) {
    IR2VEC_DEBUG(std::cout << "\tNot a load/store instruction\n");
    collectNonDepRD(inst, RD);
  } else {
    for (unsigned i = 0; i < inst->getNumOperands(); ++i) {
      llvm::Value *operand = inst->getOperand(i);
      if (!operand->getType()->isPointerTy()) {
        if (auto parent = dyn_cast<Instruction>(inst->getOperand(i))) {
          RD->push_back(parent);
        }
      }
    }
    MemoryUseOrDef *useOrDef = MSSA.getMemoryAccess(inst);
    if (useOrDef)
      populateRDWithMemssa(useOrDef, RD);
    else {
      collectNonDepRD(inst, RD);
    };
  }
}

void checkMemssaFunctions(llvm::Module &M) {
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
      MemorySSA &MSSA = FAM.getResult<MemorySSAAnalysis>(F).getMSSA();
      for (auto &BB : F) {
        for (Instruction &inst : BB) {
          llvm::SmallVector<const llvm::Instruction *, 10> RD;
          calcSSAReachingDefs(&inst, MSSA, &RD);
          if (RD.size() > 0) {
            printReachingDefs(&inst, RD);
          }
        }
      }
    }
  }
}

// void checkDepAnalysisFunctions(llvm::Module &M) {
//   PassBuilder PB;
//   FunctionAnalysisManager FAM;

//   // We need to initialize the other pass managers even if we don't directly
//   use
//   // them
//   LoopAnalysisManager LAM;
//   CGSCCAnalysisManager CGAM;
//   ModuleAnalysisManager MAM;

//   // Register all the passes with the PassBuilder
//   PB.registerModuleAnalyses(MAM);
//   PB.registerCGSCCAnalyses(CGAM);
//   PB.registerLoopAnalyses(LAM);
//   PB.registerFunctionAnalyses(FAM);

//   PB.crossRegisterProxies(LAM, FAM, CGAM, MAM);

//   // Register required alias analyses and memory dependence analysis
//   FAM.registerPass([] { return DependenceAnalysis(); });
//   FAM.registerPass([] { return BasicAA(); }); // Basic Alias Analysis

//   // Run the pass on each function in the module
//   for (Function &F : M) {
//     if (!F.isDeclaration()) {
//       DependenceInfo &result = FAM.getResult<DependenceAnalysis>(F);

//       // for (auto &BB : F) {
//       //   for (Instruction &inst : BB) {
//       //     llvm::SmallVector<const llvm::Instruction *, 10> RD;
//       //     calcSSAReachingDefs(&inst, MSSA, &RD);
//       //     if (RD.size() > 0) {
//       //       printReachingDefs(&inst, RD);
//       //     }
//       //   }
//       // }
//     }
//   }
// }

void runMDA() {
  auto M = getLLVMIR();

  // check if M is a vaid module or not
  if (!M) {
    std::cout << "Invalid module" << std::endl;
    return;
  }

  if (memdep)
    checkMemdepFunctions(*M);
  else if (memssa)
    checkMemssaFunctions(*M);

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
