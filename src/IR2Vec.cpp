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

void printWalk(Peephole &walk) {
  for (auto &inst : walk) {
    printObject(inst);
  }
}

void store_store_elimination(Peephole &walk) {
  std::cout << "Running store store elimination\n";
  std::unordered_map<Value *, unsigned> storeMap;

  for (int i = 0; i < walk.size(); i++) {
    Instruction *inst = walk[i];
    assert(dyn_cast<Instruction>(inst));

    if (auto *store = dyn_cast<StoreInst>(inst)) {
      assert(store);
      std::cout << "Got Store instruction";
      printObject(store);
      std::cout << "\n";

      Value *val = store->getPointerOperand();
      std::cout << "Got Value";
      printObject(val);
      std::cout << "\n";

      if (storeMap.find(val) != storeMap.end()) {
        // Found a store to the same address, eliminate the previouse store
        unsigned prevStore = storeMap[val];
        std::cout << "ERASING ";
        printObject(walk[prevStore]);
        walk.erase(walk.begin() + prevStore);
        std::cout << "Done erasing" << std::endl;
        assert(prevStore < i);
        i--;
      }
      storeMap[val] = i;
      std::cout << "Done storing" << std::endl;
    } else if (auto *load = dyn_cast<LoadInst>(inst)) {
      assert(load);
      std::cout << "Got Load instruction";
      printObject(load);
      std::cout << "\n";

      Value *val = load->getPointerOperand();
      std::cout << "Got load value";
      printObject(val);
      std::cout << std::endl;

      if (storeMap.find(val) != storeMap.end()) {
        std::cout << "Found a load from the same address, remove the store "
                     "instruction from the map\n";
        storeMap.erase(val);
      }
    }
  }

  if (storeMap.size() > 0) {
    std::cout << "Emptying store map" << std::endl;
    storeMap.clear();
  }
}

void normaliseWalks(std::vector<WalkSet> &FunctionWalkSet) {
  for (WalkSet &functionWalk : FunctionWalkSet) {
    for (Peephole &walk : functionWalk) {
      std::cout << "\n\nBefore store-store eliminatin\n\n";
      printWalk(walk);
      store_store_elimination(walk);
      std::cout << "\n\nAfter store-store eliminatin\n\n";
      printWalk(walk);
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

  auto M = getLLVMIR();
  auto vocabulary = VocabularyFactory::createVocabulary(DIM)->getVocabulary();

  int k = 4; // max length of random walk
  int c = 2; // min freq of each node
  std::cout << "Generating peephole set\n";

  std::vector<WalkSet> functionWalks;

  for (auto &F : *M) {
    WalkSet walks;
    generatePeepholeSet(*M, &walks, F, k, c);

    std::cout << "Walks Generated : " << walks.size() << "\n\n\n";
    if (walks.size() == 0) {
      std::cout << "No walks\n";
      continue;
    } else {
      for (auto &walk : walks) {
        printWalk(walk);
      }
    }

    functionWalks.push_back(walks);
  }

  // here - we normalize the walks
  // std::cout << "Starting normalisaton " << std::endl;
  // normaliseWalks(functionWalks);

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
