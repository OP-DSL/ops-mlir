# Standalone (no external test framework) unit tests. Each test is a
# plain executable that prints PASS/FAIL per case and exits non-zero on
# any failure; wired into `ctest` via add_test so `ctest` picks them up
# too, but they can also just be run directly.

add_executable(KernelIRBuilderTest KernelIRBuilderTest.cpp)

target_link_libraries(KernelIRBuilderTest
	PRIVATE
	OPSRuntime
)

add_test(NAME KernelIRBuilderTest COMMAND KernelIRBuilderTest)

# Pure-C++ planner tests (no JIT / GPU needed).
add_executable(FusionPlannerTest FusionPlannerTest.cpp)
target_link_libraries(FusionPlannerTest PRIVATE OPSRuntime)
add_test(NAME FusionPlannerTest COMMAND FusionPlannerTest)
