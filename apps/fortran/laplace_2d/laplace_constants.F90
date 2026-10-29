!===- laplace_constants.F90 - global constants for the 2D Laplace app ---===!
!
! Mirrors OPS_CONSTANTS in the OPS Fortran laplace2d tutorial. `pi` is a
! Fortran `parameter`, so flang folds its value into the kernels' FIR
! directly; `jmax` is an ordinary variable whose value the driver registers
! with ops_decl_const() so the JIT can bake it into the translated kernels.
!
!===----------------------------------------------------------------------===!

module OPS_CONSTANTS

    implicit none

    integer(kind=4) :: imax, jmax
    real(kind=8), parameter :: pi = 2.0_8 * asin(1.0_8)

end module OPS_CONSTANTS
