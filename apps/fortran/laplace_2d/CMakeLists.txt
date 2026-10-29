#===============================================================================
# Laplace 2D Application (Fortran)
#===============================================================================

ops_seq_backend_library()

ops_add_fortran_application(laplace_2d_ftn
  laplace2d.F90
  laplace_kernels.F90
  laplace_constants.F90)
