# bindings/CMakeLists.txt
cmake_minimum_required(VERSION 3.13)

find_package(pybind11 CONFIG REQUIRED)
pybind11_add_module(_core py_module.cpp)

# Headers from src tree (public ones)
target_include_directories(_core
  PRIVATE
    ${CMAKE_SOURCE_DIR}/src/include
    ${CMAKE_BINARY_DIR}/src/include   # where generated headers usually land
)

# Link to your IR2Vec lib target if present, and inherit its usage reqs
set(_ir2vec_tgt "")
if (TARGET ${IR2VEC_LIB})
  set(_ir2vec_tgt ${IR2VEC_LIB})
elseif (TARGET ${IR2VEC_LIB_STATIC})
  set(_ir2vec_tgt ${IR2VEC_LIB_STATIC})
endif()

if (_ir2vec_tgt)
  target_link_libraries(_core PRIVATE ${_ir2vec_tgt})

  # Ensure generation runs before compiling _core
  add_dependencies(_core ${_ir2vec_tgt})

  # Inherit the library target's include dirs & compile defs (covers generated dirs)
  target_include_directories(_core PRIVATE
    $<TARGET_PROPERTY:${_ir2vec_tgt},INCLUDE_DIRECTORIES>)
  target_compile_definitions(_core PRIVATE
    $<TARGET_PROPERTY:${_ir2vec_tgt},COMPILE_DEFINITIONS>)
else()
  message(WARNING "IR2Vec library target not found; _core may miss generated headers.")
endif()

if(UNIX AND NOT APPLE)
  target_compile_options(_core PRIVATE -fvisibility=hidden)
endif()

# Install into a Python package dir
install(TARGETS _core LIBRARY DESTINATION ir2vec)
