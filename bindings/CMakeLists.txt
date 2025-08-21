# bindings/CMakeLists.txt
cmake_minimum_required(VERSION 3.13)

find_package(pybind11 CONFIG REQUIRED)
pybind11_add_module(pr2vec py_module.cpp)

# Headers from src tree (public ones)
target_include_directories(pr2vec
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
  target_link_libraries(pr2vec PRIVATE ${_ir2vec_tgt})

  # Ensure generation runs before compiling pr2vec
  add_dependencies(pr2vec ${_ir2vec_tgt})

  # Inherit the library target's include dirs & compile defs (covers generated dirs)
  target_include_directories(pr2vec PRIVATE
    $<TARGET_PROPERTY:${_ir2vec_tgt},INCLUDE_DIRECTORIES>)
  target_compile_definitions(pr2vec PRIVATE
    $<TARGET_PROPERTY:${_ir2vec_tgt},COMPILE_DEFINITIONS>)
else()
  message(WARNING "IR2Vec library target not found; pr2vec may miss generated headers.")
endif()

if(UNIX AND NOT APPLE)
  target_compile_options(pr2vec PRIVATE -fvisibility=hidden)
endif()

# Install into a Python package dir
install(TARGETS pr2vec LIBRARY DESTINATION ir2vec)
