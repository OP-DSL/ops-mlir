# libopsjit: the runtime that plugs into OPS's lazy execution (Phase 3 fills it in). It links the OPS backend
# libraries at the application's link line, never itself.
add_library(opsjit SHARED Init.cpp)
target_include_directories(opsjit PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../include>)
set_target_properties(opsjit PROPERTIES POSITION_INDEPENDENT_CODE ON LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
