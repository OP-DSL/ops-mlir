add_library(
	OPSPasses
	UseMonoCuStream.cpp
)

target_include_directories(
	OPSPasses
	PUBLIC
	${CMAKE_SOURCE_DIR}/include
)

target_link_libraries(
	OPSPasses
	PUBLIC
	LLVMCore
	LLVMSupport
)
