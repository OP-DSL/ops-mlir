#===============================================================================
# Fortran Applications
#===============================================================================
# Built only when the Fortran frontend is enabled -- it needs an LLVM build
# with the flang project, whose frontend the JIT links to translate kernel
# bodies. See README.md.
#===============================================================================

enable_language(Fortran)
include(OPSFortranApp)

add_subdirectory(laplace_2d)
