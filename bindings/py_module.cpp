// bindings/py_module.cpp
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

// Pull in your IR2Vec public headers:
#include "IR2Vec.h"
#include "utils.h"
#include "version.h"
#include <fstream>
#include <ios>
#include <stdio.h>

namespace py = pybind11;

class IR2VecHandler {
private:
  std::string fileName;
  std::string outputFile;
  std::string mode;
  std::string level;
  IR2Vec::Embeddings *emb = nullptr;
  unsigned dim;

public:
  IR2VecHandler(std::string fileName, std::string outputFile, std::string mode,
                std::string level, unsigned dim)
      : fileName(std::move(fileName)), outputFile(std::move(outputFile)),
        mode(std::move(mode)), level(std::move(level)),
        emb(new IR2Vec::Embeddings()), dim(dim) {}

  ~IR2VecHandler() { delete emb; }

  std::string getFile() { return fileName; }
  std::string getOutputFile() { return outputFile; }
  std::string getMode() { return mode; }
  std::string getLevel() { return level; }

  IR2Vec::Vector getProgramVector() const { return emb->getProgramVector(); }

  llvm::SmallMapVector<const llvm::Instruction *, IR2Vec::Vector, 128> &
  getInstVecMap() {
    return emb->getInstVecMap();
  }

  llvm::SmallMapVector<const llvm::Function *, IR2Vec::Vector, 16> &
  getFunctionVecMap() {
    return emb->getFunctionVecMap();
  }

  bool fileNotValid(std::string filename) {
    std::ifstream temp;
    temp.open(filename, std::ios_base::in);
    if (temp.peek() == std::ifstream::traits_type::eof() ||
        temp.bad() == true || temp.fail() == true) {
      return true;
    }
    temp.close();
    return false;
  }

  void initEncodings(std::string function_name = "") {
    IR2Vec::iname = fileName;
    IR2Vec::IR2VecMode ir2vecMode =
        (mode == std::string("sym") ? IR2Vec::Symbolic : IR2Vec::FlowAware);

    std::unique_ptr<llvm::Module> Module = IR2Vec::getLLVMIR();

    emb = std::make_unique<IR2Vec::Embeddings>(*Module, ir2vecMode, level.at(0),
                                               outputFile, dim, function_name)
              .get();

    if (!emb) {
      throw std::runtime_error("Failed to create embeddings");
    }
  }
};

IR2VecHandler *createIR2VecObject(std::string filename, std::string output_file,
                                  std::string mode, std::string level,
                                  unsigned dim) {
  IR2VecHandler *ir2vecObj =
      new IR2VecHandler(filename, output_file, mode, level, dim);
  if (!ir2vecObj) {
    throw std::runtime_error("Failed to Create embeddings");
  }
  return ir2vecObj;
}

IR2VecHandler *initEmbedding(std::string filename = "", std::string mode = "",
                             std::string level = "",
                             std::string output_file = "",
                             std::string function_name = "",
                             unsigned dim = 300) {

  if (fileNotValid(filename))
    throw std::runtime_error("Invalid File Path");

  if (not output_file.empty())
    if (fileNotValid(output_file))
      throw std::runtime_error("Invalid Output File Path");

  if (not(mode == std::string("sym") or mode == std::string("fa")))
    throw std::runtime_error(
        "Eroneous mode entered . Either of sym, fa should be "
        "specified");

  if (not(level.at(0) == 'p' or level.at(0) == 'f'))
    throw std::runtime_error("Invalid level specified: Use either p or f");

  IR2VecHandler *ir2vecObj =
      createIR2VecObject(filename, output_file, mode, level, dim);

  ir2vecObj->initEncodings(function_name);

  return ir2vecObj;
}

PYBIND11_MODULE(_core, m) {
  m.doc() = R"pbdoc(
        IR2Vec Python bindings.

        Exposes selected APIs and utilities from the IR2Vec C++ library.
    )pbdoc";

  m.def(
      "getVersion", [] { return std::string(IR2VEC_VERSION); },
      "Get IR2Vec Version");

  // m.def("get_program_vector",
  //       [](const std::string& ir_path) {
  //           py::gil_scoped_release release;
  //           return getProgramVector(ir_path);   // declared in IR2Vec.h
  //       },
  //       py::arg("ir_path"),
  //       "Compute program vector for an LLVM IR file");
  //
  // m.def("get_instruction_vectors",
  //       &getInstructionVectors, py::arg("ir_path"));
}
