// bindings/py_module.cpp
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

// Pull in your IR2Vec public headers:
#include "IR2Vec.h"
#include "utils.h"

namespace py = pybind11;

PYBIND11_MODULE(_core, m) {
  m.doc() = "IR2Vec Python bindings (pybind11 skeleton)";

  // Simple canary to prove the module loads
  m.def("ping", [] { return std::string("ir2vec bindings alive"); });

  // Example: expose a version string if you have one
  // (Uncomment if you have a configured header like version.h)
  // m.def("version", [] { return std::string(IR2VEC_VERSION_STRING); });

  // TODO: add your real API. For example (adjust signatures to your API):
  //
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
