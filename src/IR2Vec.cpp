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

#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/MemoryLocation.h"
#include "llvm/Analysis/MemorySSA.h"
#include "llvm/Analysis/ValueTracking.h"
#include <llvm/Analysis/BasicAliasAnalysis.h>
#include <llvm/Analysis/DependenceAnalysis.h>

#include "llvm/IR/Instructions.h"
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

cl::opt<bool> cl_writeDefsMap("writeDefsMap", cl::Optional,
                              cl::desc("Testing writeDefsMap Collection"),
                              cl::init(false), cl::cat(category));

cl::opt<bool> cl_reachingDefsMap("reachingDefsMap", cl::Optional,
                                 cl::desc("Testing reachingDefs Collection"),
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

using MapTy =
    llvm::SmallMapVector<const llvm::Instruction *,
                         llvm::SmallVector<const llvm::Instruction *, 10>, 16>;

std::map<std::string, std::set<std::string>> normalize(const MapTy &Mp) {
  // IR2VEC_DEBUG(std::cout << "Normalize called for Map" << std::endl);
  // IR2VEC_DEBUG(std::cout << "Sanity Check . Map.size() - " << Mp.size()
  //                        << std::endl);
  std::map<std::string, std::set<std::string>> out;

  for (auto &kv : Mp) {
    // IR2VEC_DEBUG(std::cout << "Reached inside loop" << std::endl);
    std::string k = printObject(kv.first);
    // IR2VEC_DEBUG(std::cout << k << std::endl);

    std::set<std::string> vals;
    if (kv.second.empty())
      vals.insert("empty_set");
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
  if (oldMap.size() == 0 and newMap.size() == 0)
    return true;
  if (oldMap.size() == 0) {
    // IR2VEC_DEBUG(std::cout << "Size of old map is 0" << std::endl);
    return false;
  }

  if (newMap.size() == 0) {
    // IR2VEC_DEBUG(std::cout << "Size of new map is 0" << std::endl);
    return false;
  }

  // IR2VEC_DEBUG(std::cout << "Sanity Check new map size " << newMap.size()
  //                        << std::endl);
  auto newMapNorm = normalize(newMap);
  // IR2VEC_DEBUG(std::cout << "Normalize done for new map" << std::endl);

  // IR2VEC_DEBUG(std::cout << "\nSanity Check old map size " << oldMap.size()
  //                        << std::endl);
  auto oldMapNorm = normalize(oldMap);
  // IR2VEC_DEBUG(std::cout << "Normalize done for old map" << std::endl);

  return oldMapNorm == newMapNorm;
}

bool areVectorsEqual(
    const llvm::SmallVector<const llvm::Instruction *, 10> &vecA,
    const llvm::SmallVector<const llvm::Instruction *, 10> &vecB) {
  if (vecA.size() != vecB.size())
    return false;
  std::set<const llvm::Instruction *> setA(vecA.begin(), vecA.end());
  std::set<const llvm::Instruction *> setB(vecB.begin(), vecB.end());
  return setA == setB;
}

void compareMapsSimple(const MapTy oldMap, const MapTy newMap) {
  std::cout << "Old map size: " << oldMap.size()
            << ", New map size: " << newMap.size() << std::endl;

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
          std::cout << "DIFFERENT VECTORS: " << printObject(oldPair.first)
                    << std::endl;
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
            << ", Same vectors: " << sameVectors
            << ", Different vectors: " << diffVectors << std::endl;
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
  test_writeDefs = cl_writeDefsMap;
  test_reachingDefs = cl_reachingDefsMap;
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

static inline const Instruction *underlyingInst(const Instruction *I) {
  const Value *Ptr = getPointerOperand(I);
  if (!Ptr)
    return nullptr;

  if (const GetElementPtrInst *GEP = dyn_cast<GetElementPtrInst>(Ptr)) {
    return GEP;
  }

  // Get the deepest object in the pointer chain
  const Value *UnderlyingObj = llvm::getUnderlyingObject(Ptr);

  // If it's an instruction, that's our base
  if (const Instruction *BaseInst = dyn_cast<Instruction>(UnderlyingObj)) {
    return BaseInst;
  }

  // Otherwise, if the original pointer was an instruction, use that
  return dyn_cast<Instruction>(Ptr);
}

static inline const Instruction *baseInstOf(const Instruction *I) {
  if (isa<GetElementPtrInst>(I))
    return I;

  const Instruction *Base = underlyingInst(I);

  // Keep going deeper only if we can actually go deeper
  while (Base && Base->mayReadOrWriteMemory()) {
    const Instruction *NextBase = underlyingInst(Base);
    if (!NextBase || isa<GetElementPtrInst>(NextBase)) {
      // Can't go deeper, stop here
      break;
    }
    Base = NextBase;
  }

  return Base;
}

static inline void recordDefFor(MapTy &writeDefsMap,
                                const Instruction *UseOrDefInst,
                                const Instruction *DefInst) {
  const Instruction *Base = baseInstOf(UseOrDefInst);
  if (Base && DefInst)
    writeDefsMap[Base].push_back(DefInst);
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

  // TODO :: Refine this function. For now, conservative estimate - true
  return true;
}

void collectSSAWriteDefsMap(FunctionAnalysisManager &FAM, Module &M,
                            MapTy &writeDefsMap) {
  for (Function &F : M) {
    if (!F.isDeclaration()) {
      MemorySSA &MSSA = FAM.getResult<MemorySSAAnalysis>(F).getMSSA();
      for (auto &BB : F) {
        for (auto &I : BB) {
          IR2VEC_DEBUG(std::cout << "\n\nChecking instruction "
                                 << printObject(&I) << std::endl);
          if (!I.mayReadOrWriteMemory()) {
            IR2VEC_DEBUG(std::cout
                         << "Does not read of write memory. Leaving out"
                         << std::endl);
            continue;
          }
          MemoryAccess *MA = MSSA.getMemoryAccess(&I);
          if (!MA) {
            IR2VEC_DEBUG(std::cout << "Memory access not received "
                                   << std::endl);
            continue;
          }
          if (auto *MU = dyn_cast<MemoryUse>(MA)) {
            continue;
            IR2VEC_DEBUG(std::cout << "\tEntered memory Use " << std::endl);
          } else if (auto *MD = dyn_cast<MemoryDef>(MA)) {
            IR2VEC_DEBUG(std::cout << "Entered follow up branch - memDef "
                                   << std::endl);
            if (isFakeDef(&I))
              continue; // skip fake defs (volatile/atomic loads)
            recordDefFor(writeDefsMap, &I, &I);
          } else if (auto *MPhi = dyn_cast<MemoryPhi>(MA)) {
            IR2VEC_DEBUG(std::cout << "Phi node - skipping" << std::endl);
            continue;
          }
        }
      }
    }
  }
}

bool accessesSameMemoryLocation(Instruction *srcInst, Instruction *targetMem,
                                AAResults &AA) {
  if (!targetMem || !srcInst) // || !srcInst->mayWriteToMemory())
    return false;

  const Instruction *defInst = baseInstOf(srcInst);
  if (!defInst)
    defInst = srcInst;

  const Instruction *targetInst = baseInstOf(targetMem);
  if (!targetInst)
    targetInst = targetMem;

  IR2VEC_DEBUG(std::cout << "\t\t\tSrc Inst " << printObject(srcInst)
                         << " Base Inst " << printObject(defInst) << std::endl);
  IR2VEC_DEBUG(std::cout << "\t\t\ttargetMemLocation " << printObject(targetMem)
                         << " Base targetInst " << printObject(targetInst)
                         << std::endl);
  return AA.isMustAlias(defInst, targetInst);
}

void _impl_collectLiveDefinitions(MemoryAccess *DefAccess,
                                  Instruction *targetMemLocation, AAResults &AA,
                                  SmallPtrSet<const Instruction *, 32> &RD,
                                  Instruction *rootInst) {

  SmallPtrSet<const Instruction *, 32> visitedList;
  SmallVector<MemoryAccess *, 8> worklist;

  worklist.push_back(DefAccess);
  visitedList.insert(rootInst);

  // TODO : This is a heuristic
  int recurseMax = 100;

  while (!worklist.empty() && recurseMax > 0) {
    recurseMax--;
    IR2VEC_DEBUG(std::cout << "\t\tEntered worklist loop" << std::endl);
    MemoryAccess *current = worklist.pop_back_val();

    if (!current) {
      IR2VEC_DEBUG(std::cout << "\t\tCurrent is NULL - SKIPPING" << std::endl);
      continue;
    }

    IR2VEC_DEBUG(std::cout << "\t\tChecking MemAccess " << printObject(current)
                           << std::endl);

    if (auto *MD = dyn_cast<MemoryDef>(current)) {
      Instruction *defInst = MD->getMemoryInst();

      if (!defInst) {
        IR2VEC_DEBUG(std::cout << "\t\t\tdefInst is null" << std::endl);
        IR2VEC_DEBUG(std::cout << "\t\t\t Current memAccess reaching null. "
                                  "Adding targetMem Inst to RD "
                               << printObject(targetMemLocation) << std::endl;);
        RD.insert(targetMemLocation);
        continue;
      }

      if (visitedList.count(defInst) > 0) {
        IR2VEC_DEBUG(std::cout
                     << "\t\t\tInsertion Failed. Inst already visited. Skipping"
                     << std::endl);
        continue;
      }
      visitedList.insert(defInst);

      if (auto *callInst = llvm::dyn_cast<llvm::CallInst>(defInst)) {
        if (callInst->hasRetAttr(Attribute::NoAlias)) {
          // This is an allocation function - check if it matches our target
          if (accessesSameMemoryLocation(defInst, targetMemLocation, AA)) {
            RD.insert(defInst);
            continue;
          }
          IR2VEC_DEBUG(std::cout
                       << "\t\t\t\tDid not get memory Alias - moving to next"
                       << std::endl);
          worklist.push_back(MD->getDefiningAccess());
        } else {
          IR2VEC_DEBUG(std::cout
                       << "\t\t\t\tSkip this instruction - it's a function call"
                       << std::endl);
          worklist.push_back(MD->getDefiningAccess());
          continue;
        }
      }

      if (auto *LI = llvm::dyn_cast<llvm::LoadInst>(defInst);
          LI && (LI->isVolatile() || LI->isAtomic())) {
        IR2VEC_DEBUG(
            std::cout
            << "\t\t\t\tSkip this instruction - it's a volatile/Atomic load"
            << std::endl);
        worklist.push_back(MD->getDefiningAccess());
        continue;
      }

      IR2VEC_DEBUG(std::cout << "\t\t\tChecking potential memDef "
                             << printObject(defInst) << std::endl);

      if (accessesSameMemoryLocation(defInst, targetMemLocation, AA)) {
        IR2VEC_DEBUG(std::cout
                     << "\t\t\t\tEstablished Alias Memory - adding Live RD"
                     << std::endl);
        RD.insert(defInst);
        continue;
      }
      // Continue walking to find other live definitions
      IR2VEC_DEBUG(std::cout
                   << "\t\t\t\tDid not get memory Alias - moving to next"
                   << std::endl);
      worklist.push_back(MD->getDefiningAccess());
    } else if (auto *MP = dyn_cast<MemoryPhi>(current)) {
      IR2VEC_DEBUG(std::cout << "\t\tEntered memoryPhi" << std::endl);
      // Phi merges multiple live definitions
      for (unsigned i = 0; i < MP->getNumIncomingValues(); ++i) {
        worklist.push_back(MP->getIncomingValue(i));
      }
    } else {
      IR2VEC_DEBUG(std::cout
                   << "\t\tnot memory def, and not memoryPhi. Skipping"
                   << std::endl);
    }
  }
  IR2VEC_DEBUG(std::cout << "Worklist Empty - Exiting" << std::endl);
}

void startCollectLiveDefinitions(MemoryAccess *StartAccess,
                                 Instruction *targetMemLocation, AAResults &AA,
                                 SmallPtrSet<const Instruction *, 32> &RD,
                                 Instruction *rootInst) {

  if (!StartAccess) {
    IR2VEC_DEBUG(std::cout << "\t\tNo defining access found" << std::endl);
    return;
  }

  if (!targetMemLocation) {
    IR2VEC_DEBUG(std::cout << "\t\tNo target mem Location found" << std::endl);
    return;
  }
  IR2VEC_DEBUG(std::cout
               << "\t\t\tProceeding with MemorySSA analysis using StartAccess: "
               << printObject(StartAccess) << " and targetMemLocation "
               << printObject(targetMemLocation) << std::endl);

  // Walk MemorySSA chain to find all live definitions
  _impl_collectLiveDefinitions(StartAccess, targetMemLocation, AA, RD,
                               rootInst);
}

void collectMemOps(Instruction *memOperand, MemorySSA &MSSA, AAResults &AA,
                   SmallPtrSet<const Instruction *, 32> &memOpSet) {

  if (!memOperand)
    return;

  SmallPtrSet<const Value *, 32> Visited;

  std::function<void(const Value *)> collect = [&](const Value *V) {
    if (!V)
      return;
    if (!V->getType()->isPointerTy())
      return;
    if (!Visited.insert(V).second)
      return; // already visited

    // PHI: recurse on incoming values
    if (auto *PN = dyn_cast<PHINode>(V)) {
      for (unsigned i = 0, e = PN->getNumIncomingValues(); i != e; ++i)
        collect(PN->getIncomingValue(i));
      return;
    }

    // select: recurse on both arms
    if (auto *SI = dyn_cast<SelectInst>(V)) {
      collect(SI->getTrueValue());
      collect(SI->getFalseValue());
      return;
    }

    // Otherwise, if it's an Instruction pointer-producing value, add it.
    if (auto *I = dyn_cast<Instruction>(V)) {
      memOpSet.insert(I);
    }
  };

  collect(memOperand);
}

MemoryAccess *getStartAccess(Instruction *I, Instruction *memOperand,
                             MemorySSA &MSSA) {
  MemoryAccess *MA = MSSA.getMemoryAccess(I);
  if (MA) {
    // Normal case: MemoryAccess exists
    if (auto *MUOD = dyn_cast<MemoryUseOrDef>(MA)) {
      return MUOD->getDefiningAccess();
    }
  }

  IR2VEC_DEBUG(
      std::cout << "\t\tMemory access not found - handling null MemoryAccess"
                << std::endl);

  BasicBlock *BB = I->getParent();
  MemoryAccess *StartAccess = nullptr;

  // STEP 1: Look for the most recent MemoryDef BEFORE this instruction in the
  // same block
  auto *BlockAccesses = MSSA.getBlockAccesses(BB);
  if (BlockAccesses) {
    IR2VEC_DEBUG(std::cout << "\t\t\tSearching for earliest MemoryDef "
                              "predecessor instruction in same BB"
                           << std::endl);

    for (auto &Access : *BlockAccesses) {
      if (auto *MD = dyn_cast<MemoryDef>(&Access)) {
        Instruction *DefInst = MD->getMemoryInst();

        // Check if this MemoryDef comes BEFORE our instruction
        if (DefInst && DefInst->comesBefore(I)) {
          StartAccess = const_cast<MemoryDef *>(MD);
          IR2VEC_DEBUG(std::cout
                       << "\t\t\t\tFound MemoryDef before instruction: "
                       << printObject(DefInst) << std::endl);
          continue; // Keep looking for more recent MemoryDefs
        }

        // If we reach here, this MemoryDef comes after our instruction - stop
        // searching
        IR2VEC_DEBUG(
            std::cout
            << "\t\t\t\tReached MemoryDef after instruction, stopping search"
            << std::endl);
        break;
      }
    }
  }

  // STEP 2: If no MemoryDef found before instruction, check for MemoryPhi at
  // block start
  if (!StartAccess) {
    IR2VEC_DEBUG(
        std::cout
        << "\t\t\tNo MemoryDef before instruction, checking for MemoryPhi"
        << std::endl);

    if (MemoryPhi *MPhi = MSSA.getMemoryAccess(BB)) {
      StartAccess = MPhi;
      IR2VEC_DEBUG(std::cout
                   << "\t\t\t\tFound MemoryPhi for instruction's basic block"
                   << std::endl);
    }
  }

  if (!StartAccess) {
    StartAccess = MSSA.getLiveOnEntryDef();
    IR2VEC_DEBUG(
        std::cout
        << "\t\t\tNo MemoryDef found, no memoryPhi, using live-on-entry"
        << std::endl);
  }

  return StartAccess;
}

Instruction *getMemoryRoot(Instruction *memOperand) {
  if (!memOperand) {
    return nullptr;
  }

  Instruction *memRoot = llvm::findAllocaForValue(memOperand);
  if (!memRoot) {
    memRoot = memOperand;
  }

  return memRoot;
}

void collectLiveDefinitions(Instruction *I, Instruction *memOperand,
                            MemorySSA &MSSA, AAResults &AA,
                            SmallPtrSet<const Instruction *, 32> &RD) {
  if (!memOperand) {
    IR2VEC_DEBUG(std::cout << "\t\tMemory operand not found" << std::endl);
    return;
  }

  MemoryAccess *StartAccess = getStartAccess(I, memOperand, MSSA);
  Instruction *memRoot = getMemoryRoot(memOperand);

  IR2VEC_DEBUG(std::cout << "\t\t MemRoot for " << printObject(memOperand)
                         << " - " << printObject(memRoot) << std::endl);

  IR2VEC_DEBUG(std::cout << "\t\tGetting Live memory definitions for Inst "
                         << printObject(I)
                         << "\n\t\tAnd memory operand Inst is "
                         << printObject(memRoot) << std::endl);

  startCollectLiveDefinitions(StartAccess, memRoot, AA, RD, I);
}

void _impl_collectSSAReachingDefs(Instruction *I, MemorySSA &MSSA,
                                  AAResults &AA,
                                  SmallPtrSet<const Instruction *, 32> *RD) {
  RD->clear();
  IR2VEC_DEBUG(std::cout << "\n\nStudying Inst " << printObject(I)
                         << std::endl);

  if (isa<AllocaInst>(I)) {
    IR2VEC_DEBUG(std::cout << "\tAlloca Inst, return Empty RD" << std::endl);
    return;
  }

  for (unsigned opIdx = 0; opIdx < I->getNumOperands(); ++opIdx) {
    Value *operand = I->getOperand(opIdx);

    if (auto *operandInst = dyn_cast<Instruction>(operand)) {
      if (!operand->getType()->isPointerTy()) {
        // SSA Case: The operand instruction IS the definition (single def in
        // SSA)
        IR2VEC_DEBUG(std::cout << "\t Adding RD : Operand Instruction "
                               << printObject(operandInst) << std::endl);
        RD->insert(operandInst);
      } else {
        IR2VEC_DEBUG(std::cout << "\tOperand is pointer "
                               << printObject(operand) << " studying further"
                               << std::endl);
        collectLiveDefinitions(I, operandInst, MSSA, AA, *RD);
      }
    } else if (isa<Constant>(operand)) {
      IR2VEC_DEBUG(std::cout << "\tConstant value , skipping "
                             << printObject(operand) << std::endl);
      continue;
    }
  }
}

bool rejectInstCases(Instruction *I) {
  if (isa<BranchInst>(I) && !cast<BranchInst>(I)->isConditional())
    return true;
  if (isa<ReturnInst>(I) &&
      (I->getNumOperands() == 0 || isa<Constant>(I->getOperand(0))))
    return true;
  if (auto *CI = dyn_cast<CallInst>(I);
      CI && CI->getCalledFunction() && CI->getCalledFunction()->isIntrinsic())
    return true;
  return false;
}

void collectSSAReachingDefs(Instruction *I, MemorySSA &MSSA, AAResults &AA,
                            MapTy &resultMap) {
  llvm::SmallPtrSet<const Instruction *, 32> RD;
  _impl_collectSSAReachingDefs(I, MSSA, AA, &RD);
  if (!rejectInstCases(I)) {
    resultMap[I].assign(RD.begin(), RD.end());
  }
}

void collectSSAReachingDefs_wrapper(FunctionAnalysisManager &FAM, Module &M,
                                    MapTy &resultMap) {
  // Run the pass on each function in the module
  for (Function &F : M) {
    if (!F.isDeclaration()) {
      MemorySSA &MSSA = FAM.getResult<MemorySSAAnalysis>(F).getMSSA();
      // AAManager::Result models AAResults
      AAResults &AA = FAM.getResult<AAManager>(F);
      for (auto &BB : F) {
        for (Instruction &inst : BB) {
          collectSSAReachingDefs(&inst, MSSA, AA, resultMap);
        }
      }
    }
  }
}

// SmallMapVector<const Instruction*, SmallVector<const Instruction*, 10>, 16>
void checkMemssaFunctions(llvm::Module &M, MapTy &resultMap) {
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

  if (IR2Vec::test_writeDefs) {
    if (IR2Vec::printTime) {
      IR2Vec::timeFunction("SSA collectWriteDefs map", [&]() {
        collectSSAWriteDefsMap(FAM, M, resultMap);
      });
    } else
      collectSSAWriteDefsMap(FAM, M, resultMap);
  }

  else if (IR2Vec::test_reachingDefs) {
    if (IR2Vec::printTime) {
      IR2Vec::timeFunction("SSA ReachingDefs map", [&]() {
        collectSSAReachingDefs_wrapper(FAM, M, resultMap);
      });
    } else
      collectSSAReachingDefs_wrapper(FAM, M, resultMap);
  }
}

void test_writedefsmap() {
  auto M = getLLVMIR();
  if (!M) {
    std::cout << "Invalid module" << std::endl;
    return;
  }
  auto vocabulary = VocabularyFactory::createVocabulary(DIM)->getVocabulary();

  IR2Vec_FA FA(*M, vocabulary);
  auto oldMap = FA.getWriteDefsMap();

  // std::cout << "Old Map is ready" << std::endl;
  // IR2Vec::print_write_defs_map(oldMap);
  MapTy newMap;
  checkMemssaFunctions(*M, newMap);
  // std::cout << "New Map Ready " << std::endl;
  // IR2Vec::print_write_defs_map(newMap);

  // compareMapsSimple(oldMap, newMap);
  bool same = writeDefsMapEqualByText(oldMap, newMap);
  std::cout << "Both maps are Same ? - " << same << std::endl;
}

void test_reachingdefs() {
  auto M = getLLVMIR();

  // check if M is a vaid module or not
  if (!M) {
    std::cout << "Invalid module" << std::endl;
    return;
  }

  // get old Map / Old Defs
  // if(!IR2Vec::debug)
  // auto newMap = checkMemssaFunctions(*M);
  // std::cout << "New Map Ready " << std::endl;
  // IR2Vec::print_write_defs_map(newMap);
  // new Reaching Defs

  llvm::SmallMapVector<const llvm::Instruction *,
                       llvm::SmallVector<const llvm::Instruction *, 10>, 16>
      oldReachingDefs;
  // compareMapsSimple(oldMap, newMap);
  // bool same = writeDefsMapEqualByText(oldMap, newMap);
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
  IR2VEC_DEBUG(for (auto &Inst
                    : oldReachingDefs) {
    auto RD = Inst.second;
    auto inst = Inst.first;
    IR2Vec::printReachingDefs(inst, RD);
  });
  IR2VEC_DEBUG(std::cout << "==> Native Reaching Defs Finished" << std::endl);

  IR2VEC_DEBUG(std::cout << "\n\nPrinting SSA Reaching Defs" << std::endl);
  MapTy newReachingDefs;
  checkMemssaFunctions(*M, newReachingDefs);
  IR2VEC_DEBUG(std::cout << "\n\nPrinting Final SSA Reaching Defs\n\n"
                         << std::endl);

  IR2VEC_DEBUG(for (auto &Inst
                    : newReachingDefs) {
    auto RD = Inst.second;
    auto inst = Inst.first;
    IR2Vec::printReachingDefs(inst, RD);
  });
  IR2VEC_DEBUG(std::cout << "==> SSA Reaching Defs Finished" << std::endl);
  // sanity check

  // std::cout << "\n\n\n Sanity Check - Old Reaching Defs Normalize" <<
  // std::endl; if(!IR2Vec::debug) { auto oldNormalizedMap =
  // normalize(oldReachingDefs); auto newNormalizedMap =
  // normalize(newReachingDefs); bool same = (oldNormalizedMap ==
  // newNormalizedMap);

  // auto oldReachingDefs = generateFAEncodings();
  IR2VEC_DEBUG(std::cout << "Both Reaching Defs Ready, starting comparison"
                         << std::endl);
  IR2VEC_DEBUG(compareMapsSimple(oldReachingDefs, newReachingDefs));
  bool same = writeDefsMapEqualByText(oldReachingDefs, newReachingDefs);
  std::cout << "Both maps are Same ? - " << same << std::endl;
  // }
  // new Reaching Defs
  // std::cout << "\n\n Printing SSA Reaching Defs" << std::endl;
  // auto newReachingDefs = checkMemssaFunctions(*M);
  // std::cout << "\n\n New SSA Reaching Defs ready" << std::endl;s
}

void runMDA() {
  if (IR2Vec::test_writeDefs)
    test_writedefsmap();
  else if (IR2Vec::test_reachingDefs)
    test_reachingdefs();
  else
    std::cout << "Please specify either of -writeDefsMap / -reachingDefsMap";

  return;
}

int main(int argc, char **argv) {
  cl::SetVersionPrinter(printVersion);
  cl::HideUnrelatedOptions(category);
  setGlobalVars(argc, argv);
  checkFailureConditions();

  runMDA();
  return 0;
}
