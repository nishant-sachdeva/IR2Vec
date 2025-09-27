//===- utils.cpp - Helper utilities  ---------------------------*- C++ -*-===//
//
// Part of the IR2Vec Project, under the Apache License v2.0 with LLVM
// Exceptions. See the LICENSE file for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "utils.h"
#include "IR2Vec.h"
#include "Vocabulary.h"
#include <fstream>
#include <iostream>
#include <sstream> // for std::stringstream
#include <string>

using namespace llvm;
using namespace IR2Vec;

bool IR2Vec::fa;
bool IR2Vec::sym;
bool IR2Vec::printTime;
bool IR2Vec::collectIR;
std::string IR2Vec::iname;
std::string IR2Vec::oname;
std::string IR2Vec::funcName;
char IR2Vec::level;
int IR2Vec::cls;
float IR2Vec::WO;
float IR2Vec::WA;
float IR2Vec::WT;
bool IR2Vec::debug;
unsigned IR2Vec::DIM;
bool IR2Vec::test_writeDefs;
bool IR2Vec::test_reachingDefs;

void IR2Vec::_impl_collectLiveDefinitions_Walker(
    MemoryAccess *DefAccess, Instruction *targetMemLocation, AAResults &AA,
    SmallPtrSet<const Instruction *, 32> &RD, Instruction *rootInst,
    MemorySSA &MSSA) {

  SmallPtrSet<MemoryAccess *, 32> visitedList;
  SmallVector<MemoryAccess *, 8> worklist;
  MemorySSAWalker *Walker = MSSA.getWalker();
  // MemoryLocation OperandLoc =
  // MemoryLocation().getWithNewPtr(targetMemLocation);
  MemoryLocation OperandLoc =
      MemoryLocation(targetMemLocation, LocationSize::precise(1));
  // MemoryLocation OperandLoc(
  //   targetMemLocation,
  //   LocationSize::precise(
  //     (targetMemLocation->getModule()->getDataLayout())
  //     .getTypeStoreSize(targetMemLocation->getType())
  //   )
  // );

  IR2VEC_DEBUG(std::cout << "Memory Location Operand loc found is "
                         << printObject(&OperandLoc) << std::endl);
  // default Definition sets - LocationSize::beforeOrAfterPointer());

  IR2VEC_DEBUG(std::cout << "\tSanity Check Live on Entry "
                         << printObject(MSSA.getLiveOnEntryDef()) << std::endl);

  worklist.push_back(DefAccess);
  // visitedList.insert(DefAccess);

  // TODO : This is a heuristic
  int recurseMax = 100;

  while (!worklist.empty() && recurseMax > 0) {
    recurseMax--;
    IR2VEC_DEBUG(std::cout << "\t\tEntered worklist loop" << std::endl);
    MemoryAccess *current = worklist.pop_back_val();

    if (!current) {
      IR2VEC_DEBUG(std::cout << "\t\tCurrent memAccess is NULL - SKIPPING"
                             << std::endl);
      continue;
    }

    IR2VEC_DEBUG(std::cout << "\t\tChecking MemAccess " << printObject(current)
                           << std::endl);

    MemoryAccess *ClobberingAccess =
        Walker->getClobberingMemoryAccess(current, OperandLoc);

    if (visitedList.count(ClobberingAccess) > 0) {
      IR2VEC_DEBUG(std::cout << "\t\t\tInsertion Failed. ClobberingAccess "
                                "already visited. Skipping"
                             << std::endl);
      continue;
    }
    visitedList.insert(ClobberingAccess);

    IR2VEC_DEBUG(std::cout << "\t\tChecking ClobberingAccess "
                           << printObject(ClobberingAccess) << std::endl);

    if (auto *MD = dyn_cast<MemoryDef>(ClobberingAccess)) {
      IR2VEC_DEBUG(std::cout << "\t\tChecking MD " << printObject(MD)
                             << std::endl);

      if (MSSA.isLiveOnEntryDef(MD)) {
        IR2VEC_DEBUG(std::cout << "\t\t\t Current memAccess is Live On Def "
                                  "Adding targetMem Inst to RD "
                               << printObject(targetMemLocation) << std::endl;);
        RD.insert(targetMemLocation);
        continue;
      }

      Instruction *defInst = MD->getMemoryInst();

      if (!defInst) {
        IR2VEC_DEBUG(std::cout
                         << "\t\t\t Current memAccess-defInst reaching null. "
                            "Adding targetMem Inst to RD "
                         << printObject(targetMemLocation) << std::endl;);
        RD.insert(targetMemLocation);
        continue;
      }

      IR2VEC_DEBUG(std::cout << "\t\tFetched DefInst " << printObject(defInst) << std::endl);

      if (auto *callInst = llvm::dyn_cast<llvm::CallInst>(defInst)) {
        IR2VEC_DEBUG(std::cout << "\t\t Studying call Inst" << std::endl);
        Function* calledInst = callInst->getCalledFunction();
        IR2VEC_DEBUG(std::cout << "\t\t Fetched getCalledFunction" << std::endl);
        if(!calledInst) {
          worklist.push_back(MD->getDefiningAccess());
          continue;
        }
        bool isIntr = calledInst->isIntrinsic();
        IR2VEC_DEBUG(std::cout << "\t\t Fetched isIntrinSic" << std::endl);
        if (isIntr) {
          IR2VEC_DEBUG(std::cout << "\t\t\t\t Internal Library Call ( malloc etc)"
                                 << std::endl);
          RD.insert(callInst);
          continue;
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
        IR2VEC_DEBUG(std::cout << "\t\t\tgetClobberingAccess() terminated at a "
                                  "volatile/Atomic load. Search Further"
                               << std::endl);
        worklist.push_back(MD->getDefiningAccess());
        continue;
      }

      if (defInst == rootInst) {
        IR2VEC_DEBUG(std::cout << "Loop Circle - Don't add rootInst"
                               << std::endl);
        continue;
      }

      IR2VEC_DEBUG(std::cout << "\t\t\tAdding MemDef Live Clobbering RD "
                             << printObject(defInst) << std::endl);

      // IR2VEC_DEBUG(std::cout
      //              << "\t\t\t\tFor the Record. Current Alias Function - Gives "
      //              << std::string(IR2Vec::accessesSameMemoryLocation(
      //                                 defInst, targetMemLocation, AA)
      //                                 ? "True"
      //                                 : "False")
      //              << std::endl);

      RD.insert(defInst);
    } else if (auto *MP = dyn_cast<MemoryPhi>(ClobberingAccess)) {
      IR2VEC_DEBUG(std::cout << "\t\tEntered memoryPhi" << std::endl);
      // Phi merges multiple live definitions
      for (unsigned i = 0; i < MP->getNumIncomingValues(); ++i) {
        auto childMP = MP->getIncomingValue(i);
        IR2VEC_DEBUG(std::cout << "\t\t\tLogging Child memoryPhi "
                               << printObject(childMP) << std::endl);
        worklist.push_back(childMP);
      }
    } else {
      IR2VEC_DEBUG(std::cout
                   << "\t\tnot memory def, and not memoryPhi. Potential Error"
                   << std::endl);
    }
  }
  IR2VEC_DEBUG(std::cout << "Worklist Empty - Exiting" << std::endl);
}

void IR2Vec::startCollectLiveDefinitions(MemoryAccess *StartAccess,
                                 Instruction *targetMemLocation, AAResults &AA,
                                 SmallPtrSet<const Instruction *, 32> &RD,
                                 Instruction *rootInst, MemorySSA &MSSA) {

  if (!StartAccess) {
    IR2VEC_DEBUG(std::cout << "\t\tNo defining access found. Insert RD and back"
                           << std::endl);
    RD.insert(IR2Vec::getMemoryRoot(targetMemLocation));
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
  // _impl_collectLiveDefinitions(StartAccess, targetMemLocation, AA, RD,
  //                              rootInst);

  IR2Vec::_impl_collectLiveDefinitions_Walker(StartAccess, targetMemLocation, AA, RD,
                                      rootInst, MSSA);
}

MemoryAccess* IR2Vec::getStartAccess(Instruction *I, Instruction *memOperand,
                             MemorySSA &MSSA) {
  MemoryAccess *MA = MSSA.getMemoryAccess(I);
  if (MA) {
    IR2VEC_DEBUG(std::cout << "\t\tNormal case: MemoryAccess exists"
                           << std::endl);
    if (auto *MUOD = dyn_cast<MemoryUseOrDef>(MA)) {
      return MUOD->getDefiningAccess();
    }
  }

  IR2VEC_DEBUG(
      std::cout << "\t\tMemory access not found - returning BB MemoryAccess"
                << std::endl);

  BasicBlock *BB = I->getParent();
  // return MSSA.getMemoryAccess(BB);
  // }
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

  return StartAccess;
}

void IR2Vec::collectLiveDefinitions(Instruction *I, Instruction *memOperand,
                            MemorySSA &MSSA, AAResults &AA,
                            SmallPtrSet<const Instruction *, 32> &RD) {
  if (!memOperand) {
    IR2VEC_DEBUG(std::cout << "\t\tMemory operand not found" << std::endl);
    return;
  }

  MemoryAccess *StartAccess = IR2Vec::getStartAccess(I, memOperand, MSSA);
  // Instruction *memRoot = IR2Vec::getMemoryRoot(memOperand);
  Instruction *memRoot = memOperand;

  IR2VEC_DEBUG(std::cout << "\t\t MemRoot for " << printObject(memOperand)
                         << " - " << printObject(memRoot) << std::endl);

  IR2VEC_DEBUG(std::cout << "\t\tGetting Live memory definitions for Inst "
                         << printObject(I)
                         << "\n\t\tAnd memory operand Inst is "
                         << printObject(memRoot) << std::endl);

  IR2Vec::startCollectLiveDefinitions(StartAccess, memRoot, AA, RD, I, MSSA);
}

void IR2Vec::_impl_collectSSAReachingDefs(Instruction *I, MemorySSA &MSSA,
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
        IR2VEC_DEBUG(std::cout << "\tSSA Operand. Direct Insert "
                               << printObject(operand) << std::endl);
        RD->insert(operandInst);
      } else {
        IR2VEC_DEBUG(std::cout << "\tOperand is pointer "
                               << printObject(operand) << " studying further"
                               << std::endl);
        IR2Vec::collectLiveDefinitions(I, operandInst, MSSA, AA, *RD);
      }
    } else if (isa<Constant>(operand)) {
      IR2VEC_DEBUG(std::cout << "\tConstant value , skipping "
                             << printObject(operand) << std::endl);
      continue;
    }
  }
}


void IR2Vec::collectSSAReachingDefs(Instruction *I, MemorySSA &MSSA, AAResults &AA,
                            IR2Vec::MapTy &resultMap) {
  llvm::SmallPtrSet<const Instruction *, 32> RD;
  IR2Vec::_impl_collectSSAReachingDefs(I, MSSA, AA, &RD);
  if (!IR2Vec::rejectInstCases(I)) {
    resultMap[I].assign(RD.begin(), RD.end());
  }
}

Instruction* IR2Vec::getMemoryRoot(Instruction *memOperand) {
  if (!memOperand) {
    return nullptr;
  }

  Instruction *memRoot = llvm::findAllocaForValue(memOperand);
  if (!memRoot) {
    memRoot = memOperand;
  }

  return memRoot;
}

bool IR2Vec::accessesSameMemoryLocation(Instruction *srcInst, Instruction *targetMem,
                                AAResults &AA) {
  if (!targetMem || !srcInst) // || !srcInst->mayWriteToMemory())
    return false;

  const Instruction *defInst = IR2Vec::baseInstOf(srcInst);
  if (!defInst)
    defInst = srcInst;

  const Instruction *targetInst = IR2Vec::baseInstOf(targetMem);
  if (!targetInst)
    targetInst = targetMem;

  // IR2VEC_DEBUG(std::cout << "\t\t\tSrc Inst " << printObject(srcInst)
  //                        << " Base Inst " << printObject(defInst) <<
  //                        std::endl);
  // IR2VEC_DEBUG(std::cout << "\t\t\ttargetMemLocation " <<
  // printObject(targetMem)
  //                        << " Base targetInst " << printObject(targetInst)
  //                        << std::endl);
  return AA.isMustAlias(defInst, targetInst);
  // return !AA.isNoAlias(defInst, targetInst);
}

bool IR2Vec::rejectInstCases(llvm::Instruction *I) {
  if (isa<llvm::BranchInst>(I) && !cast<llvm::BranchInst>(I)->isConditional())
    return true;
  if (isa<llvm::ReturnInst>(I) &&
      (I->getNumOperands() == 0 || isa<llvm::Constant>(I->getOperand(0))))
    return true;
  if (auto *CI = dyn_cast<llvm::CallInst>(I);
      CI && CI->getCalledFunction() && CI->getCalledFunction()->isIntrinsic())
    return true;
  return false;
}


std::unique_ptr<llvm::Module> IR2Vec::getLLVMIR() {
  static llvm::LLVMContext context;
  SMDiagnostic err;
  auto M = parseIRFile(iname, err, context);

  if (!M) {
    err.print(iname.c_str(), outs());
    exit(1);
  }

  return M;
}

void IR2Vec::print_write_defs_map(
  llvm::SmallMapVector<const llvm::Instruction *,
                       llvm::SmallVector<const llvm::Instruction *, 10>, 16>
      writeDefsMapObj
) {
  std::cout << writeDefsMapObj.size() << std::endl;

  std::cout << "Printing Write Defs Map " << std::endl;
  for(auto item : writeDefsMapObj) {
    auto ins = item.first;
    auto defs = item.second;

    // print out instruction
    std::cout << IR2Vec::printObject(ins) << "\t";
    if(defs.size() > 0){
      for(auto def_ins : defs) {
        std::cout << "\t" << IR2Vec::printObject(def_ins);
      }
      std::cout << std::endl;
    }
  }
  std::cout << "Write Defs Map concluded" << std::endl;
}


std::string IR2Vec::getInstStr(const llvm::Instruction* inst) {
  std::string useStr;
  llvm::raw_string_ostream useStream(useStr);

  inst->print(useStream);
  return useStream.str();
}

void IR2Vec::printDependency(const llvm::Instruction* use, const llvm::Instruction* def) {
  std::cout << IR2Vec::getInstStr(use) << " dependent on " << IR2Vec::getInstStr(def) << std::endl;
}

bool IR2Vec::isLoad(const llvm::Instruction* I) {
  return (std::string(I->getOpcodeName()) == "load") ? true : false;
}

bool IR2Vec::isStore(const llvm::Instruction* I) {
  return (std::string(I->getOpcodeName()) == "store") ? true : false;
}

bool IR2Vec::isLoadorStore(const llvm::Instruction* I) {
  return IR2Vec::isLoad(I) || IR2Vec::isStore(I);
}

void IR2Vec::printReachingDefs(const llvm::Instruction *I, llvm::SmallVector<const llvm::Instruction*, 10> RD) {
  std::cout << IR2Vec::getInstStr(I) << " dependent on";

  for (auto reachInst : RD) {
    std::cout <<  " " << IR2Vec::getInstStr(reachInst);
  }

  std::cout << std::endl;
}

void IR2Vec::scaleVector(Vector &vec, float factor) {
  for (unsigned i = 0; i < vec.size(); i++) {
    vec[i] = vec[i] * factor;
  }
}

// Function to get demangled function name
std::string IR2Vec::getDemagledName(const llvm::Function *function) {
  auto functionName = function->getName().str();
  std::size_t sz = 17;
  int status;
  char *const readable_name =
      __cxa_demangle(functionName.c_str(), 0, &sz, &status);
  auto demangledName =
      status == 0 ? std::string(readable_name) : std::string(functionName);
  free(readable_name);
  return demangledName;
}

// Function to get actual function name
char *IR2Vec::getActualName(llvm::Function *function) {
  auto functionName = function->getName().str();
  auto demangledName = getDemagledName(function);
  size_t Size = 1;
  char *Buf = static_cast<char *>(std::malloc(Size));
  const char *mangled = functionName.c_str();
  char *baseName;
  llvm::ItaniumPartialDemangler Mangler;
  if (Mangler.partialDemangle(mangled)) {
    baseName = &demangledName[0];
  } else {
    baseName = Mangler.getFunctionBaseName(Buf, &Size);
  }
  return baseName;
}

// Function to return updated res
std::string IR2Vec::updatedRes(IR2Vec::Vector tmp, llvm::Function *f,
                               llvm::Module *M) {
  std::string res = "";
  auto demangledName = getDemagledName(f);

  res += M->getSourceFileName() + "__" + demangledName + "\t";

  res += "=\t";
  for (auto i : tmp) {
    if ((i <= 0.0001 && i > 0) || (i < 0 && i >= -0.0001)) {
      i = 0;
    }
    res += std::to_string(i) + "\t";
  }

  return res;
}
