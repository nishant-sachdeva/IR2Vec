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

#include "llvm/IR/Function.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Transforms/InstCombine/InstCombine.h"
#include "llvm/Transforms/Scalar/SimplifyCFG.h"
#include "llvm/Transforms/Utils/Mem2Reg.h"

#include <iostream>
#include <stdio.h>
#include <time.h>

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

template <typename T> void printObject(const T *obj) {
  if (!obj) {
    std::cout << "Null object - no object to print\n";
    return;
  }
  std::string output;
  llvm::raw_string_ostream rso(output);
  obj->print(rso); // Call the `print` method of the object
  rso.flush();
  std::cout << output << std::endl;
}

#include <random> // For random number generation

// Function to generate a random number in the range [0, x)
int getRandomNumber(int x) {
  if (x <= 0) {
    throw std::invalid_argument("Input x must be greater than 0");
  }

  // Seed the random number generator
  std::random_device rd;  // Non-deterministic random device
  std::mt19937 gen(rd()); // Mersenne Twister RNG

  // Define the range [0, x)
  std::uniform_int_distribution<> distrib(0, x - 1);

  // Generate a random number
  return distrib(gen);
}

// define type PeepHole = vector<Instructions>
using Peephole = std::vector<llvm::Instruction *>;
using WalkSet = std::vector<Peephole>;

// Define the struct to hold the embeddings
struct FunctionWalkEmbeddings {
  std::vector<std::string>
      functionStrings; // collect global strings for each function
  std::vector<std::string>
      functionCalls; // functionNames for function calls made in the walkSet
  std::string functionName; // Corresponding to functionName
  WalkSet walkSet;          // Corresponding to walkSet

  IR2Vec::opcodeEmbedding opcodeEmb;   // Corresponding to opcodeEmbedding
  IR2Vec::typeEmbedding typeEmb;       // Corresponding to typeEmbedding
  IR2Vec::operandEmbedding operandEmb; // Corresponding to operandEmbedding
};

void printWalk(Peephole &walk) {
  for (auto &inst : walk) {
    printObject(inst);
  }
}

void resolvePhiInst(Peephole &walk) {
  // std::cout << "Resolving phi instructions\n";
  for (int i = 0; i < walk.size(); i++) {
    Instruction *inst = walk[i];
    assert(dyn_cast<Instruction>(inst));

    if (PHINode *phiObj = dyn_cast<PHINode>(inst)) {
      if (phiObj->getNumIncomingValues() == 0) {
        continue;
      }

      Value *selectedValue = phiObj->getIncomingValue(0);

      printObject(phiObj);
      std::cout << "\tphiObj value\t";
      printObject(selectedValue);

      LLVMContext &context = phiObj->getContext();
      IRBuilder<> builder(context);

      Instruction *newInst = nullptr;

      // Check if the selectedValue is a pointer
      if (selectedValue->getType()->isPointerTy()) {
        // Create a load instruction for pointer types
        newInst = dyn_cast<Instruction>(builder.CreateLoad(
            selectedValue->getType(), selectedValue, phiObj->getName()));
      } else {
        // Create a direct scalar assignment (in LLVM, this is effectively an
        // alias or move)
        auto *assignInst =
            builder.CreateLoad(selectedValue->getType(), selectedValue);
        assignInst->setName(phiObj->getName());
        newInst = dyn_cast<Instruction>(assignInst);
      }

      std::cout << "\tnewInst\t";
      printObject(newInst);
      std::cout << std::endl;

      walk[i] = std::move(newInst);
    }
  }
}

void remove_unconditional_branches(Peephole &walk) {
  // std::cout << "Running unconditional branch elimination\n";
  for (int i = 0; i < walk.size(); i++) {
    Instruction *inst = walk[i];
    assert(dyn_cast<Instruction>(inst));

    if (auto *br = dyn_cast<BranchInst>(inst)) {
      assert(br);
      if (br->isUnconditional()) {
        // std::cout << "ERASING branch Instruction \t"; printObject(br);
        walk.erase(walk.begin() + i);
        i--;
      }
    }
  }
}

void remove_type_extensions(Peephole &walk) {
  for (int i = 0; i < walk.size(); i++) {
    Instruction *inst = walk[i];
    assert(dyn_cast<Instruction>(inst));

    if (isa<ZExtInst>(inst) || isa<SExtInst>(inst) || isa<FPExtInst>(inst) ||
        isa<FPTruncInst>(inst) || isa<TruncInst>(inst)) {
      walk.erase(walk.begin() + i);
      i--;
    }
  }
}

void store_store_elimination(Peephole &walk) {
  // std::cout << "Running store store elimination\n";
  std::unordered_map<Value *, unsigned> storeMap;

  for (int i = 0; i < walk.size(); i++) {
    Instruction *inst = walk[i];
    assert(dyn_cast<Instruction>(inst));

    if (auto *store = dyn_cast<StoreInst>(inst)) {
      assert(store);
      Value *val = store->getPointerOperand();
      if (storeMap.find(val) != storeMap.end()) {
        unsigned prevStore = storeMap[val];
        walk.erase(walk.begin() + prevStore);
        assert(prevStore < i);
        i--;
      }
      storeMap[val] = i;
    } else if (auto *load = dyn_cast<LoadInst>(inst)) {
      assert(load);
      Value *val = load->getPointerOperand();
      if (storeMap.find(val) != storeMap.end()) {
        storeMap.erase(val);
      }
    }
  }

  if (storeMap.size() > 0) {
    // std::cout << "Emptying store map" << std::endl;
    storeMap.clear();
  }
}

void load_load_elimination(Peephole &walk) {
  /*
    I1 : %x = load %9
    ...
    I2 : %y = load %9

    1. if between i1 and i2, there are no instructions that change %9
    then I2 is a redundant load
    2. If we don't replace all follow up occurrences of %y - we loose flow
    information
                TODO:: for now, it is acceptable since we are only taking
    symbolic values
  */
  std::unordered_map<Value *, unsigned> storeMap;

  for (int i = 0; i < walk.size(); i++) {
    Instruction *inst = walk[i];
    assert(dyn_cast<Instruction>(inst));

    if (auto *loadInst = dyn_cast<LoadInst>(inst)) {
      assert(loadInst);
      Value *val = loadInst->getPointerOperand();
      if (storeMap.find(val) != storeMap.end()) {
        walk.erase(walk.begin() + i);
        i--;
      } else {
        storeMap[val] = i;
      }
    } else if (auto *storeInst = dyn_cast<StoreInst>(inst)) {
      assert(storeInst);
      Value *val = storeInst->getPointerOperand();
      if (storeMap.find(val) != storeMap.end()) {
        storeMap.erase(val);
      }
    }
  }

  if (storeMap.size() > 0) {
    // std::cout << "Emptying store map" << std::endl;
    storeMap.clear();
  }
}

void normaliseFunctionWalks(
    std::vector<FunctionWalkEmbeddings> &FunctionWalkSet) {
  for (FunctionWalkEmbeddings &functionWalk : FunctionWalkSet) {
    for (Peephole &walk : functionWalk.walkSet) {
      // std::cout << "\n\nBefore store-store eliminatin\n\n";
      // printWalk(walk);
      store_store_elimination(walk);
      load_load_elimination(walk);
      remove_unconditional_branches(walk);
      // resolvePhiInst(walk);
      remove_type_extensions(walk);
      // std::cout << "\n\nAfter store-store eliminatin\n\n";
      // printWalk(walk);
    }
  }
}

void addBBToWalk(Peephole &walk, BasicBlock *r1) {
  // std::cout << "Adding BB to walk\n";
  for (auto &Inst : *r1) {
    walk.push_back(&Inst);
  }
}

void generatePeepholeSet(llvm::Module &M, WalkSet *walks, llvm::Function &F,
                         int k, int c) {
  std::cout << "Function: " << F.getName().data() << "\n";
  std::vector<BasicBlock *> bbset, starters;
  std::unordered_map<BasicBlock *, int> visited;

  for (auto &BB : F) {
    starters.push_back(&BB);
    visited[&BB] = 0;
  }

  while (starters.size() > 0) {
    // std::cout << "Starters: " << starters.size() << "\n";

    int idx = getRandomNumber(starters.size());
    BasicBlock *r1 = starters[idx];
    if (!r1) {
      std::cout << "No starter\n";
      break;
    }

    int walk_len = 0;
    Peephole walk;
    while (walk_len <= k) {
      // std::cout << "\tWalk: " << walk_len << "\n";
      addBBToWalk(walk, r1);
      visited[r1]++;
      walk_len++;

      if (visited[r1] >= c) {
        auto index = std::find(starters.begin(), starters.end(), r1);
        if (index != starters.end()) {
          starters.erase(index);
        }
      }

      unsigned numSuccessors = llvm::succ_size(r1);
      if (numSuccessors == 0) {
        // std::cout << "No successors\n";
        break;
      } else {
        idx = getRandomNumber(numSuccessors);
        auto successors = llvm::successors(r1);
        bbset = std::vector<BasicBlock *>(successors.begin(), successors.end());
        // std::cout << "Successors: " << bbset.size() << "\n";
        r1 = bbset[idx];
        if (!r1) {
          std::cout << "No successor - ERROR\n";
          break;
        }
      }
    }
    walks->push_back(walk);
  }
  return;
}

void runPassesOnModule(llvm::Module &M) {
  llvm::PassBuilder PB;
  llvm::FunctionAnalysisManager FAM;
  llvm::ModuleAnalysisManager MAM;
  llvm::LoopAnalysisManager LAM;
  llvm::CGSCCAnalysisManager CGAM;

  PB.registerFunctionAnalyses(FAM);
  PB.registerModuleAnalyses(MAM);
  PB.registerCGSCCAnalyses(CGAM);
  PB.registerLoopAnalyses(LAM);
  PB.crossRegisterProxies(LAM, FAM, CGAM, MAM);

  llvm::FunctionPassManager FPM;
  FPM.addPass(llvm::PromotePass());
  FPM.addPass(llvm::InstCombinePass());
  FPM.addPass(llvm::SimplifyCFGPass());

  llvm::ModulePassManager MPM;
  MPM.addPass(llvm::createModuleToFunctionPassAdaptor(std::move(FPM)));
  MPM.run(M, MAM);
}

void printFunctionWalks(std::vector<FunctionWalkEmbeddings> &functionWalks) {
  for (auto &functionWalk : functionWalks) {
    std::cout << "\n\n\n\nwalks generated - " << functionWalk.walkSet.size()
              << "\n";
    for (auto walk : functionWalk.walkSet) {
      std::cout << "\n\nWalk: \n";
      printWalk(walk);
    }
  }
}

void removeTypeCastsFromModule(llvm::Module &M) {
  for (auto &F : M) {
    for (auto &BB : F) {
      for (auto it = BB.begin(); it != BB.end();) {
        Instruction *I = &*it++;

        if (auto *cast = dyn_cast<ZExtInst>(I)) {
          Value *val = cast->getOperand(0);
          cast->replaceAllUsesWith(val);
          cast->eraseFromParent();
        }
      }
    }
  }
}

void runCustomPassesOnModule(llvm::Module &M) { removeTypeCastsFromModule(M); }

template <typename T>
void addEmbeddingVecs(std::vector<T> &vec1, std::vector<T> &vec2) {
  std::transform(vec1.begin(), vec1.end(), vec2.begin(), vec1.begin(),
                 std::plus<T>());
}

void generateSymbolicWalkEmbeddings(FunctionWalkEmbeddings &functionWalk,
                                    IR2Vec_Symbolic &SYM) {
  for (Peephole &walk : functionWalk.walkSet) {
    std::vector<int> embedding;
    opcodeEmbedding opcode_embedding;
    typeEmbedding type_embedding;
    operandEmbedding operand_embedding;

    for (Instruction *inst : walk) {
      SYM.getInstructionEmbeddingsTup(inst, opcode_embedding, type_embedding,
                                      operand_embedding);
    }

    addEmbeddingVecs(functionWalk.opcodeEmb, opcode_embedding);
    addEmbeddingVecs(functionWalk.typeEmb, type_embedding);
    addEmbeddingVecs(functionWalk.operandEmb, operand_embedding);
  }

  std::cout << "Instruction Embedding tuple done for "
            << functionWalk.functionName << std::endl;
  return;
}

void calculateSymbolicWalkEmbeddings(
    std::vector<FunctionWalkEmbeddings> &functionWalks, IR2Vec_Symbolic &SYM) {
  for (auto &functionWalk : functionWalks) {
    generateSymbolicWalkEmbeddings(functionWalk, SYM);
  }
}

std::vector<std::string> getStringsFromFunction(llvm::Function &F) {
  std::vector<std::string> strList;
  for (auto &BB : F) {
    for (auto &Inst : BB) {
      for (unsigned idx = 0; idx < Inst.getNumOperands(); idx++) {
        if (auto *val =
                llvm::dyn_cast<llvm::GlobalVariable>(Inst.getOperand(idx))) {
          if (val->hasInitializer()) {
            if (auto *cda = llvm::dyn_cast<llvm::ConstantDataArray>(
                    val->getInitializer())) {
              if (cda->isString()) {
                std::string str = cda->getAsString().str();
                strList.push_back(str);
              }
            }
          }
        }
      }
    }
  }
  return strList;
}

void getSymbolicEmbeddingSet(llvm::Module &M, IR2Vec::VocabTy &vocab) {
  int k = 4; // max length of random walk
  int c = 2; // min freq of each node
  std::cout << "Generating peephole set\n";

  // M.print(outs(), nullptr);
  runPassesOnModule(M);
  IR2Vec_Symbolic SYM(M, vocab);
  // runCustomPassesOnModule(M);
  // M.print(outs(), nullptr);

  std::vector<FunctionWalkEmbeddings> functionWalkEmbeddings;
  // std::unordered_map<std::string, FunctionWalkEmbeddings> functionWalkMap;

  for (auto &F : M) {
    WalkSet walks;
    std::vector<std::string> strList = getStringsFromFunction(F);

    std::cout << "strList for " << F.getName().data() << " is "
              << strList.size() << std::endl;

    generatePeepholeSet(M, &walks, F, k, c);

    FunctionWalkEmbeddings functionWalkEmbeddingObj;
    std::string functionName = F.getName().data();
    functionWalkEmbeddingObj.functionName = functionName;
    functionWalkEmbeddingObj.walkSet = walks;
    functionWalkEmbeddingObj.functionStrings = strList;
    functionWalkEmbeddings.push_back(functionWalkEmbeddingObj);

    // functionWalkMap[functionName] = functionWalkEmbeddingObj;
  }

  // printFunctionWalks(functionWalks);

  // here - we normalize the walks
  std::cout << "Starting normalisaton " << std::endl;
  normaliseFunctionWalks(functionWalkEmbeddings);

  printFunctionWalks(functionWalkEmbeddings);

  calculateSymbolicWalkEmbeddings(functionWalkEmbeddings, SYM);
}

int main(int argc, char **argv) {
  cl::SetVersionPrinter(printVersion);
  cl::HideUnrelatedOptions(category);
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

  // bool failed = false;
  // if (!((sym ^ fa) ^ collectIR)) {
  //   errs() << "Either of sym, fa or collectIR should be specified\n";
  //   failed = true;
  // }

  // if (sym || fa) {
  //   if (level != 'p' && level != 'f') {
  //     errs() << "Invalid level specified: Use either p or f\n";
  //     failed = true;
  //   }
  // } else {
  //   if (!collectIR) {
  //     errs() << "Either of sym, fa or collectIR should be specified\n";
  //     failed = true;
  //   } else if (level)
  //     errs() << "[WARNING] level would not be used in collectIR mode\n";
  // }

  // if (failed)
  //   exit(1);

  std::unique_ptr<llvm::Module> M = getLLVMIR();

  auto vocabulary = VocabularyFactory::createVocabulary(DIM)->getVocabulary();

  getSymbolicEmbeddingSet(*M, vocabulary);

  // newly added
  // if (sym && !(funcName.empty())) {
  //   IR2Vec_Symbolic SYM(*M, vocabulary);
  //   std::ofstream o;
  //   o.open(oname, std::ios_base::app);
  //   if (printTime) {
  //     clock_t start = clock();
  //     SYM.generateSymbolicEncodingsForFunction(&o, funcName);
  //     clock_t end = clock();
  //     double elapsed = double(end - start) / CLOCKS_PER_SEC;
  //     printf("Time taken by on-demand generation of symbolic encodings "
  //            "is: %.6f "
  //            "seconds.\n",
  //            elapsed);
  //   } else {
  //     SYM.generateSymbolicEncodingsForFunction(&o, funcName);
  //   }
  //   o.close();
  // } else if (fa && !(funcName.empty())) {
  //   IR2Vec_FA FA(*M, vocabulary);
  //   std::ofstream o, missCount, cyclicCount;
  //   o.open(oname, std::ios_base::app);
  //   missCount.open("missCount_" + oname, std::ios_base::app);
  //   cyclicCount.open("cyclicCount_" + oname, std::ios_base::app);
  //   if (printTime) {
  //     clock_t start = clock();
  //     FA.generateFlowAwareEncodingsForFunction(&o, funcName, &missCount,
  //                                              &cyclicCount);
  //     clock_t end = clock();
  //     double elapsed = double(end - start) / CLOCKS_PER_SEC;
  //     printf("Time taken by on-demand generation of flow-aware encodings "
  //            "is: %.6f "
  //            "seconds.\n",
  //            elapsed);
  //   } else {
  //     FA.generateFlowAwareEncodingsForFunction(&o, funcName, &missCount,
  //                                              &cyclicCount);
  //   }
  //   o.close();
  // } else if (fa) {
  //   IR2Vec_FA FA(*M, vocabulary);
  //   std::ofstream o, missCount, cyclicCount;
  //   o.open(oname, std::ios_base::app);
  //   missCount.open("missCount_" + oname, std::ios_base::app);
  //   cyclicCount.open("cyclicCount_" + oname, std::ios_base::app);
  //   if (printTime) {
  //     clock_t start = clock();
  //     FA.generateFlowAwareEncodings(&o, &missCount, &cyclicCount);
  //     clock_t end = clock();
  //     double elapsed = double(end - start) / CLOCKS_PER_SEC;
  //     printf("Time taken by normal generation of flow-aware encodings "
  //            "is: %.6f "
  //            "seconds.\n",
  //            elapsed);
  //   } else {
  //     FA.generateFlowAwareEncodings(&o, &missCount, &cyclicCount);
  //   }
  //   o.close();
  // } else if (sym) {
  //   IR2Vec_Symbolic SYM(*M, vocabulary);
  //   std::ofstream o;
  //   o.open(oname, std::ios_base::app);
  //   if (printTime) {
  //     clock_t start = clock();
  //     SYM.generateSymbolicEncodings(&o);
  //     clock_t end = clock();
  //     double elapsed = double(end - start) / CLOCKS_PER_SEC;
  //     printf("Time taken by normal generation of symbolic encodings is: "
  //            "%.6f "
  //            "seconds.\n",
  //            elapsed);
  //   } else {
  //     SYM.generateSymbolicEncodings(&o);
  //   }
  //   o.close();
  // } else if (collectIR) {
  //   CollectIR cir(M);
  //   std::ofstream o;
  //   o.open(oname, std::ios_base::app);
  //   cir.generateTriplets(o);
  //   o.close();
  // }

  return 0;
}
