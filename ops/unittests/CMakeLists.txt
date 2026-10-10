# Unit tests of the OPS-specific libraries. They build ops.chain with ChainBuilder.h and run the analyses.
add_opsjit_unittest(ops_footprint_test FootprintTest.cpp LINK OPSMLIRAnalysis MLIRArithDialect)
add_opsjit_unittest(ops_chain_dependence_test ChainDependenceTest.cpp LINK OPSMLIRAnalysis MLIRArithDialect)
add_opsjit_unittest(ops_fusion_planner_test FusionPlannerTest.cpp LINK OPSMLIRAnalysis MLIRArithDialect)
