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

#include "llvm/Analysis/MemorySSA.h"
#include "llvm/Analysis/AliasAnalysis.h" 
#include "llvm/Analysis/MemoryLocation.h"
#include "llvm/Analysis/ValueTracking.h"
#include <llvm/Analysis/BasicAliasAnalysis.h>
#include <llvm/Analysis/DependenceAnalysis.h>

#include "llvm/Support/CommandLine.h"
#include <llvm/Analysis/MemoryDependenceAnalysis.h>
#include <llvm/IR/BasicBlock.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/Instruction.h>
#include "llvm/IR/Instructions.h"

#include <llvm/IR/Module.h>
#include <llvm/Passes/PassBuilder.h>
#include <llvm/Support/raw_ostream.h>

#include "llvm/Passes/PassPlugin.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Transforms/Scalar.h"


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

using MapTy = llvm::SmallMapVector<
    const llvm::Instruction*,
    llvm::SmallVector<const llvm::Instruction*, 10>, 16>;

std::map<std::string, std::set<std::string>> normalize(const MapTy &Mp) {
  // IR2VEC_DEBUG(std::cout << "Normalize called for Map" << std::endl);
  // IR2VEC_DEBUG(std::cout << "Sanity Check . Map.size() - " << Mp.size() << std::endl);
  std::map<std::string, std::set<std::string>> out;

  for (auto &kv : Mp) {
    // IR2VEC_DEBUG(std::cout << "Reached inside loop" << std::endl);
    std::string k = printObject(kv.first);
    // IR2VEC_DEBUG(std::cout << k << std::endl);

    std::set<std::string> vals;
    if(kv.second.empty()) vals.insert("empty_set");
    else {
      for (const llvm::Instruction *v : kv.second) {
        std::string obj = printObject(v);
        // IR2VEC_DEBUG(std::cout << "\t\t" << obj << std::endl);
        vals.insert(obj);
      }
    }
    // IR2VEC_DEBUG(std::cout << "Vals.size " << vals.size() << std::endl);
    out.emplace(std::move(k), std::move(vals));
  }
  return out;
}

bool writeDefsMapEqualByText(const MapTy &oldMap, const MapTy &newMap) {
  // IR2VEC_DEBUG(std::cout << "Entered function to compare maps" << std::endl);
  if(oldMap.size() == 0) {
    // IR2VEC_DEBUG(std::cout << "Size of old map is 0" << std::endl);
    return false;
  }

  if (newMap.size() == 0) {
    // IR2VEC_DEBUG(std::cout << "Size of new map is 0" << std::endl);
    return false;
  }

  // IR2VEC_DEBUG(std::cout << "Sanity Check new map size " << newMap.size() << std::endl);
  auto newMapNorm = normalize(newMap);
  // IR2VEC_DEBUG(std::cout << "Normalize done for new map" << std::endl);
  
  // IR2VEC_DEBUG(std::cout << "\nSanity Check old map size " << oldMap.size() << std::endl);
  auto oldMapNorm = normalize(oldMap);
  // IR2VEC_DEBUG(std::cout << "Normalize done for old map" << std::endl);

  return oldMapNorm == newMapNorm;
}

bool areVectorsEqual(const llvm::SmallVector<const llvm::Instruction*, 10>& vecA, 
                     const llvm::SmallVector<const llvm::Instruction*, 10>& vecB) {
  if (vecA.size() != vecB.size()) return false;
  std::set<const llvm::Instruction*> setA(vecA.begin(), vecA.end());
  std::set<const llvm::Instruction*> setB(vecB.begin(), vecB.end());
  return setA == setB;
}

void compareMapsSimple(const MapTy oldMap, const MapTy newMap) {
  std::cout << "Old map size: " << oldMap.size() << ", New map size: " << newMap.size() << std::endl;
  
  // Count missing and extra keys
  int missingCount = 0, extraCount = 0, sameVectors = 0, diffVectors = 0;
  
  for (const auto &oldPair : oldMap) {
    bool found = false;
    for (const auto &newPair : newMap) {
      if (oldPair.first == newPair.first) {
        found = true;
        // Compare vectors for common keys
        if (areVectorsEqual(oldPair.second, newPair.second)) {
          sameVectors++;
        } else {
          diffVectors++;
          std::cout << "DIFFERENT VECTORS: " << printObject(oldPair.first) << std::endl;
        }
        break;
      }
    }
    if (!found) {
      std::cout << "MISSING: " << printObject(oldPair.first) << std::endl;
      missingCount++;
    }
  }
  
  for (const auto &newPair : newMap) {
    bool found = false;
    for (const auto &oldPair : oldMap) {
      if (newPair.first == oldPair.first) {
        found = true;
        break;
      }
    }
    if (!found) {
      std::cout << "EXTRA: " << printObject(newPair.first) << std::endl;
      extraCount++;
    }
  }
  
  std::cout << "Missing: " << missingCount << ", Extra: " << extraCount 
            << ", Same vectors: " << sameVectors << ", Different vectors: " << diffVectors << std::endl;
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

// SmallMapVector<const Instruction*, SmallVector<const Instruction*,10>,16> 
IR2Vec_FA generateFAEncodings() {
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

  return FA;

  // print Reaching Defs
  // auto reachingDefs = FA.getInstReachingDefsMap();
  // std::cout << "\n\n Native Reaching Defs ready\n";
  // for (auto &Inst: reachingDefs) {
  //   auto RD = Inst.second;
  //   auto inst = Inst.first;
  //   IR2Vec::printReachingDefs(inst, RD);
  // }

  // // sanity check

  // std::cout << "\n\n\n Sanity Check - Old Reaching Defs Normalize" << std::endl;
  // auto normalizedMap = normalize(reachingDefs);

  // return reachingDefs;

  // std::cout << "Old Map is ready" << std::endl;
  // auto oldMap = FA.getWriteDefsMap()
  // IR2Vec::print_write_defs_map(oldMap);
  // std::cout << "\n\n";
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

void printOperand(llvm::Value *operand) {
  std::cout << "Operand: ";
  IR2Vec::printObject(operand);

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

  assert(dyn_cast<LoadInst>(inst) || dyn_cast<StoreInst>(inst));

  auto value = inst->getOperand(0);
  if (auto *parent = dyn_cast<Instruction>(value)) {
    Visited[parent] = true;
    RD->push_back(parent);
  }

  // if (auto store = dyn_cast<StoreInst>(inst)) {
  //   auto value = store->getValueOperand();
  //   if (auto *parent = dyn_cast<Instruction>(value)) {
  //     Visited[parent] = true;
  //     RD->push_back(parent);
  //   }
  // }
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


static inline const Instruction* baseInstOf(const Instruction *I) {
  const Value *Ptr = getPointerOperand(I);
  if (!Ptr) 
    return nullptr;

  // Get the deepest object in the pointer chain
  const Value *UnderlyingObj = llvm::getUnderlyingObject(Ptr);
  
  // If it's an instruction, that's our base
  if (const Instruction *BaseInst = dyn_cast<Instruction>(UnderlyingObj)) {
    return BaseInst;
  }

  // Corner case: if getUnderlyingObject stopped at a non-instruction,
  // but the original pointer was a GEP, check the GEP's base pointer
  if (const GetElementPtrInst *GEP = dyn_cast<GetElementPtrInst>(Ptr)) {
    const Value *GEPBase = GEP->getPointerOperand();
    if (const Instruction *GEPBaseInst = dyn_cast<Instruction>(GEPBase)) {
      return GEPBaseInst;
    }
  }
  
  // Otherwise, if the original pointer was an instruction, use that
  return dyn_cast<Instruction>(Ptr);
}

static inline void recordDefFor(
    SmallMapVector<const Instruction*, SmallVector<const Instruction*,10>,16>
        &writeDefsMap,
    const Instruction *UseOrDefInst,
    const Instruction *DefInst) {
      const Instruction* Base = baseInstOf(UseOrDefInst);

      // Keep going deeper only if we can actually go deeper
      while(Base && Base->mayReadOrWriteMemory()) {
        const Instruction* NextBase = baseInstOf(Base);
        if (!NextBase) {
            // Can't go deeper, stop here
            break;
        }
        Base = NextBase;
      }  

      if(Base && DefInst) writeDefsMap[Base].push_back(DefInst);
}

inline bool isFakeDef(const llvm::Instruction *I) {
  if (isa<llvm::StoreInst, llvm::AtomicRMWInst, llvm::AtomicCmpXchgInst>(I))
    return false;

  if (auto *MI = dyn_cast<llvm::MemIntrinsic>(I)) {
    // memcpy, memmove, memset all write
    return false;
  }

  // Loads never count as writes, even if atomic/volatile
  if (isa<llvm::LoadInst>(I))
    return true;

  return true;
}

llvm::SmallMapVector<const llvm::Instruction *,
                       llvm::SmallVector<const llvm::Instruction *, 10>, 16> writeDefsMap;
  
// llvm::SmallMapVector<const llvm::Instruction *,
//                        llvm::SmallVector<const llvm::Instruction *, 10>, 16> 
void collectSSAWriteDefsMap(FunctionAnalysisManager &FAM, Module &M) {
  // std::cout << "Inside SSA writeDefsMap " << std::endl;

  for (Function &F: M) {
    if (!F.isDeclaration()) {
      MemorySSA &MSSA = FAM.getResult<MemorySSAAnalysis>(F).getMSSA();
      for (auto &BB : F) {
        for (auto &I : BB) {
          // std::cout << "\n\nChecking instruction ";
          // IR2Vec::printObject(&I);
          // std::cout << "isvolatile ? " << isVolatileOrAtomic(&I) << std::endl;
          if (!I.mayReadOrWriteMemory()) continue;
            // std::cout << "Does not read of write memory. Leaving out\n";
            // printObject(&I);

            // const Instruction* Base = baseInstOf(&I);
            // std::cout << " For reference, Base here is ";
            // if(Base) printObject(Base); else std::cout << "Null " << std::endl;

          // if (isExcludedMemoryOp(&I))
          //   continue;
          MemoryAccess *MA = MSSA.getMemoryAccess(&I);
          if (!MA) {
            // std::cout << "Memory access not received " << std::endl;
            continue;
          }
          if (auto *MU = dyn_cast<MemoryUse>(MA)) {
            continue;
              // std::cout << "\tEntered memory Use " << std::endl;
            // MemorySSAWalker *Walker = MSSA.getWalker();
            // MemoryAccess *Def = Walker->getClobberingMemoryAccess(MU);
            // if (auto *MDef = dyn_cast<MemoryDef>(Def)) {
            //   std::cout << "\tFound memory Def for use " << std::endl;
            //   Instruction *DefInst = MDef->getMemoryInst();

            //   recordDefFor(writeDefsMap, &I, DefInst);
            // }
          } else if (auto *MD = dyn_cast<MemoryDef>(MA)) {
            // std::cout << "Entered follow up branch - memDef " << std::endl;
            // if (isFakeDef(&I))
            if (isa<llvm::LoadInst>(&I))
              continue; // skip fake defs (volatile/atomic loads)
            recordDefFor(writeDefsMap, &I, &I);
          } else if (auto *MPhi = dyn_cast<MemoryPhi>(MA)) {
            // std::cout << "Phi node - skipping for now" << std::endl;
            continue;
          }
        }
      }
    }
  }
  // return writeDefsMap;
}


// bool accessesSameMemoryLocation(Instruction *defInst, Value *targetMem, AAResults &AA) {
//   // Check if defInst modifies the same memory location as 
//   if(!targetMem) {
//     IR2VEC_DEBUG(
//       std::cout << "\t\tTargetMem is Null " << std::endl
//     );
//   }
//   IR2VEC_DEBUG(
//     std::cout << "\t\t\tChecking memory alias" << std::endl
//   );

//   if (auto *store = dyn_cast<StoreInst>(defInst)) {
//     Value *storePtr = store->getPointerOperand();
//     return AA.isMustAlias(storePtr, targetMem);
//   }
  
//   if (auto *load = dyn_cast<LoadInst>(defInst)) {
//     Value *loadPtr = load->getPointerOperand();
//     return AA.isMustAlias(loadPtr, targetMem);
//   }

//   if (isa<AllocaInst>(defInst)) {
//     return defInst == targetMem;
//   }

//   IR2VEC_DEBUG(
//     std::cout << "\t\tGetting memory location and Alias" << std::endl
//   );
  
//   if (defInst->mayWriteToMemory()) {
//     IR2VEC_DEBUG(
//       std::cout << "\t\t defInst reads or write memory" << std::endl
//     );
//     MemoryLocation defLoc = MemoryLocation::get(defInst);
//     if (!defLoc.Ptr) {
//       IR2VEC_DEBUG(
//         std::cout << "defloc ptr is null" << std::endl
//       );
//       return false;
//     }
//     IR2VEC_DEBUG(
//       std::cout << "\t\t defLoc fetched " << printObject(defLoc.Ptr) << std::endl
//     );
//     MemoryLocation targetLoc(targetMem, LocationSize::beforeOrAfterPointer());
//     if(!targetLoc.Ptr) {
//       IR2VEC_DEBUG(
//         std::cout << "\t\t targetLoc ptr is null" << std::endl
//       );
//       return false;
//     }
//     IR2VEC_DEBUG(
//       std::cout << "\t\t TargetLoc fetched " << printObject(targetLoc.Ptr) << std::endl
//     );
//     auto aliasresult = AA.alias(defLoc, targetLoc);
//     IR2VEC_DEBUG(
//       std::cout << "\t\t Alias Result fetched" << std::endl
//     );
//     return aliasresult != AliasResult::NoAlias;
//   }

//   return false;
// }

bool accessesSameMemoryLocation(Instruction *defInst, Value *targetMem, AAResults &AA) {
  IR2VEC_DEBUG(
    std::cout << "\t\t\tChecking Alias between " << printObject(defInst) << " and " << printObject(targetMem) << std::endl
  );

  if(!targetMem) {
    IR2VEC_DEBUG(std::cout << "\t\tTargetMem is Null " << std::endl);
    return false;
  }
  
  MemoryLocation targetLoc(targetMem, LocationSize::beforeOrAfterPointer());
  if(!targetLoc.Ptr) {
    IR2VEC_DEBUG(std::cout << "\t\t targetLoc ptr is null" << std::endl);
    return false;
  }

  // Handle alloca specially (pointer comparison)
  if (isa<AllocaInst>(defInst)) {
    return defInst == targetMem;
  }

  // For all memory-accessing instructions, use general alias analysis
  if(defInst->mayReadOrWriteMemory()) {
    MemoryLocation defLoc = MemoryLocation::get(defInst);
    if (!defLoc.Ptr) {
      IR2VEC_DEBUG(std::cout << "\t\t\tdefLoc pointer is Null" << std::endl);
      return false;
    }
    
    auto aliasResult = AA.alias(defLoc, targetLoc);
    bool result = (aliasResult != AliasResult::NoAlias);
    IR2VEC_DEBUG(std::cout << "\t\t\tALias is " << result << std::endl);
    return result;
  }

  IR2VEC_DEBUG(std::cout << "\t\t\tdefInst does not read/write memory. ALias is false" << std::endl);
  return false;
}

void collectLiveDefinitions(MemoryAccess *DefAccess, Value *targetMemLocation,
                           AAResults &AA, SmallVector<const Instruction*, 10> &RD,
                          Instruction* rootInst) {

  // SmallPtrSet<Instruction*, 8> visited;
  SmallVector<const Instruction *, 100> visitedList;
  SmallVector<MemoryAccess*, 8> worklist;

  worklist.push_back(DefAccess);
  visitedList.push_back(rootInst);

  int recurseMax = 100;

  while(!worklist.empty() && recurseMax>0) {
    recurseMax--;
    IR2VEC_DEBUG(
      std::cout << "\t\tEntered worklist loop" << std::endl
    );
    MemoryAccess *current = worklist.pop_back_val();
  
    if (!current){
      IR2VEC_DEBUG(std::cout << "\t\tCurrent is NULL - SKIPPING" << std::endl);
      continue;
    }

    IR2VEC_DEBUG(std::cout << "\t\tChecking MemAccess " << printObject(current) << std::endl);

    // std::cout << "\t\t - Checking Def types" << std::endl;
    if (auto *MD = dyn_cast<MemoryDef>(current)) {
      Instruction *defInst = MD->getMemoryInst();
      if(!defInst) {
        IR2VEC_DEBUG(std::cout << "\t\t\tdefInst is null" << std::endl);
        if(auto *rootInst = dyn_cast<Instruction>(targetMemLocation)) {
          IR2VEC_DEBUG(
            std::cout << "\t\t\t Current memAccess reaching null. target mem is Inst. Adding to RD" << std::endl;
          );
          RD.push_back(rootInst);
        }
        continue;
      }

      if (std::find(visitedList.begin(), visitedList.end(), defInst) !=
                  visitedList.end()) {
        IR2VEC_DEBUG(
          std::cout << "\t\t\tInsertion Failed. Inst already visited. Skipping" << std::endl
        );
        continue;
      }
      visitedList.push_back(defInst);

      if (llvm::isa<llvm::CallInst>(defInst)) {
        IR2VEC_DEBUG(std::cout << "\t\t\t\tSkip this instruction - it's a function call" << std::endl);
        worklist.push_back(MD->getDefiningAccess());
        continue;
      }

      if (llvm::isa<llvm::LoadInst>(defInst)) {
        IR2VEC_DEBUG(std::cout << "\t\t\t\tSkip this instruction - it's a volatile load" << std::endl);
        worklist.push_back(MD->getDefiningAccess());
        continue;
      }

      IR2VEC_DEBUG(std::cout << "\t\t\tChecking potential memDef " << printObject(defInst) << std::endl);

      if (accessesSameMemoryLocation(defInst, targetMemLocation, AA)) {
        IR2VEC_DEBUG(std::cout << "\t\t\t\tEstablished Alias Memory - adding Live RD" << std::endl);
        RD.push_back(defInst);
        continue;
      }
      // Continue walking to find other live definitions
      IR2VEC_DEBUG(std::cout << "\t\t\t\t Did not get memory Alias - moving to next" << std::endl);
      worklist.push_back(MD->getDefiningAccess());
    }
    else if (auto *MP = dyn_cast<MemoryPhi>(current)) {
      IR2VEC_DEBUG(std::cout << "\tEntered memoryPhi" << std::endl);
      // Phi merges multiple live definitions
      for (unsigned i = 0; i < MP->getNumIncomingValues(); ++i) {
        worklist.push_back(MP->getIncomingValue(i));
      }
    }
    else {
      IR2VEC_DEBUG(std::cout << "not memory def, and not memoryPhi" << std::endl);
    }
  }
  IR2VEC_DEBUG(std::cout << "Worklist Empty - Exiting" << std::endl);
}

// Value* getMemoryOperand(Instruction *I) {
//   // Use LLVM's built-in MemoryLocation API to get memory operands
//   IR2VEC_DEBUG(std::cout << "\t\tGetting memory operands for " << printObject(I) << std::endl);
//   if (!I->mayReadOrWriteMemory()) {
//     IR2VEC_DEBUG(std::cout << "\t\tDoes not read or write memory" << std::endl);
//     return nullptr;
//   }
  
//   // Try to get a specific memory location for this instruction
//   MemoryLocation loc = MemoryLocation::get(I);
//   if (loc.Ptr) {
//     return const_cast<Value*>(loc.Ptr);
//   }
  
//   // For alloca, it creates a memory location (itself)
//   // if (isa<AllocaInst>(I)) {
//   //   IR2VEC_DEBUG(std::cout << "\t\tAlloca instruction. Return as is" << std::endl);
//   //   return I;
//   // }
  
//   // For instructions that don't have a single memory location
//   // (like calls with multiple memory effects), return nullptr
//   return nullptr;
// }


void getLiveMemoryDefinitions(Instruction *I, Value* memOperand, MemorySSA &MSSA, AAResults &AA,
                             SmallVector<const Instruction*, 10> &RD) {
  // Get the memory operand for this instruction
  // Value *memOperand = getMemoryOperand(I);
  if (!memOperand) {
    IR2VEC_DEBUG(std::cout << "\t\tMemory operand not found" << std::endl);
    return;
  }

  IR2VEC_DEBUG(
    std::cout << "\t\tGetting Live memory definitions for Inst " << printObject(I) 
    << "\n\t\tAnd memory operand is " << printObject(memOperand) << std::endl
  );
  
  MemoryAccess *MA = MSSA.getMemoryAccess(I);
  if (!MA) {
    IR2VEC_DEBUG(std::cout << "\t\tMemory access not found" << std::endl);
    if(isa<Instruction>(memOperand)) {
      IR2VEC_DEBUG(
        std::cout << "\t\t memAccess is Null, but memOperand is Instr Inst. Adding" << std::endl
      );
      RD.push_back(dyn_cast<Instruction>(memOperand));
    }
    return;
  }
  
  MemoryAccess *DefiningAccess = nullptr;
  if (auto *MU = dyn_cast<MemoryUse>(MA)) {
    DefiningAccess = MU->getDefiningAccess();
  }
  else if (auto *MD = dyn_cast<MemoryDef>(MA)) {
    DefiningAccess = MD->getDefiningAccess();
  }
  
  if (!DefiningAccess) return;

  // Walk MemorySSA chain to find all live definitions
  collectLiveDefinitions(DefiningAccess, memOperand, AA, RD, I);
}

void calcSSAReachingDefs_Curr(Instruction *I, MemorySSA &MSSA,  AAResults &AA, 
                        SmallVector<const Instruction*, 10> *RD) {
  RD->clear();
  IR2VEC_DEBUG(std::cout << "\n\nStudying Inst " << printObject(I) << std::endl);

  if(isa<AllocaInst>(I)) {
    IR2VEC_DEBUG(std::cout << "\tAlloca Inst, return Empty RD" << std::endl);
    return; 
  }

  for (unsigned opIdx = 0; opIdx < I->getNumOperands(); ++opIdx) {
    Value *operand = I->getOperand(opIdx);

    if (auto *operandInst = dyn_cast<Instruction>(operand)) {
      if (!operand->getType()->isPointerTy()) { 
        // SSA Case: The operand instruction IS the definition (single def in SSA)
        IR2VEC_DEBUG(std::cout << "\t Adding RD : Operand Instruction " << printObject(operandInst) << std::endl);
        RD->push_back(operandInst);
      } else {
        // if (I->mayReadOrWriteMemory()) {
          IR2VEC_DEBUG(std::cout << "\tMay read of write memory, studying further" << std::endl);
          getLiveMemoryDefinitions(I, operand, MSSA, AA, *RD);
        // }

      }
    } else if (isa<Constant>(operand)) {
      IR2VEC_DEBUG(std::cout << "\tConstant value , skipping " << printObject(operand) << std::endl);
      continue;
    }
  }
}


SmallMapVector<const Instruction*, SmallVector<const Instruction*, 10>, 16>
checkMemssaFunctions(llvm::Module &M) {
  // std::cout << "Calling MemorySSA Functions" << std::endl;
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
  FAM.registerPass([] { return TargetLibraryAnalysis(); });

  // Install a proper AA stack (BasicAA + CFLAA + ScopedNoAliasAA, etc.)
  FAM.registerPass([&] { return PB.buildDefaultAAPipeline(); });

  // clock_t start = clock();

  // // auto writeDefsMap = 
  // collectSSAWriteDefsMap(FAM, M);

  // clock_t end = clock();
  // double elapsed = double(end - start) / CLOCKS_PER_SEC;
  // printf("Time taken by SSA collectWriteDefs map "
  //         "is: %.6f "
  //         "seconds.\n",
  //         elapsed);


  // return writeDefsMap;
  // writeDefsMap is global variable, available for access
  SmallMapVector<const Instruction*, SmallVector<const Instruction*, 10>, 16> reachingDefsMap;

  // Run the pass on each function in the module
  for (Function &F : M) {
    if (!F.isDeclaration()) {
      MemorySSA &MSSA = FAM.getResult<MemorySSAAnalysis>(F).getMSSA();

      // AAManager::Result models AAResults
      AAResults &AA = FAM.getResult<AAManager>(F);
      for (auto &BB : F) {
        for (Instruction &inst : BB) {
          llvm::SmallVector<const llvm::Instruction *, 10> RD;
          // calcSSAReachingDefs(&inst, MSSA, &RD);
          calcSSAReachingDefs_Curr(&inst, MSSA, AA, &RD);
          if(RD.size() > 0 || isa<AllocaInst>(&inst) || 
            isa<CallInst>(&inst) ||
            isa<GetElementPtrInst>(&inst) ||
            isa<ICmpInst>(&inst)) {
            reachingDefsMap[&inst] = RD;
          }
        }
      }
    }
  }
  return reachingDefsMap;
}

void runMDA() {
  auto M = getLLVMIR();

  // check if M is a vaid module or not
  if (!M) {
    std::cout << "Invalid module" << std::endl;
    return;
  }

  if (memdep)
    checkMemdepFunctions(*M);
  else if (memssa) {
    // get old Map / Old Defs
    // if(!IR2Vec::debug)
    // auto newMap = checkMemssaFunctions(*M);
    // std::cout << "New Map Ready " << std::endl;
    // IR2Vec::print_write_defs_map(newMap);
    // new Reaching Defs
    
    llvm::SmallMapVector<const llvm::Instruction *, llvm::SmallVector<const llvm::Instruction *, 10>, 16> oldReachingDefs;
    // compareMapsSimple(oldMap, newMap);
    // bool same = writeDefsMapEqualByText(oldMap, newMap);
    auto M = getLLVMIR();
    auto vocabulary = VocabularyFactory::createVocabulary(DIM)->getVocabulary();

    IR2Vec_FA FA(*M, vocabulary);
    std::ofstream o, missCount, cyclicCount;
    o.open(oname, std::ios_base::app);
    missCount.open("missCount_" + oname, std::ios_base::app);
    cyclicCount.open("cyclicCount_" + oname, std::ios_base::app);
    IR2Vec::debug = false;
    FA.generateFlowAwareEncodings(&o, &missCount, &cyclicCount);
    o.close();
    IR2Vec::debug = cl_debug;

    oldReachingDefs = FA.getInstReachingDefsMap();
    IR2VEC_DEBUG(std::cout << "Native Reaching Defs ready" << std::endl);
    IR2VEC_DEBUG(
      for (auto &Inst: oldReachingDefs) {
        auto RD = Inst.second;
        auto inst = Inst.first;
        IR2Vec::printReachingDefs(inst, RD);
      }
    );
    IR2VEC_DEBUG(std::cout << "==> Native Reaching Defs Finished" << std::endl);

    IR2VEC_DEBUG(std::cout << "\n\nPrinting SSA Reaching Defs" << std::endl);
    auto newReachingDefs = checkMemssaFunctions(*M);
    IR2VEC_DEBUG(std::cout << "\n\nPrinting Final SSA Reaching Defs\n\n" << std::endl);

    IR2VEC_DEBUG(
      for (auto &Inst: newReachingDefs) {
        auto RD = Inst.second;
        auto inst = Inst.first;
        IR2Vec::printReachingDefs(inst, RD);
      }
    );
    IR2VEC_DEBUG(std::cout << "==> SSA Reaching Defs Finished" << std::endl);
    // sanity check

    // std::cout << "\n\n\n Sanity Check - Old Reaching Defs Normalize" << std::endl;
    // if(!IR2Vec::debug) {
      // auto oldNormalizedMap = normalize(oldReachingDefs);
      // auto newNormalizedMap = normalize(newReachingDefs);
      // bool same = (oldNormalizedMap == newNormalizedMap);

      // auto oldReachingDefs = generateFAEncodings();
    IR2VEC_DEBUG(
      std::cout << "Both Reaching Defs Ready, starting comparison" << std::endl
    );
    compareMapsSimple(oldReachingDefs, newReachingDefs);
    bool same = writeDefsMapEqualByText(oldReachingDefs, newReachingDefs);
    std::cout << "Both maps are Same ? - " << same << std::endl;
    // }
    // new Reaching Defs
    // std::cout << "\n\n Printing SSA Reaching Defs" << std::endl;
    // auto newReachingDefs = checkMemssaFunctions(*M);
    // std::cout << "\n\n New SSA Reaching Defs ready" << std::endl;s
  }

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
