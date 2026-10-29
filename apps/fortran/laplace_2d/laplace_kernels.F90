!===- laplace_kernels.F90 - Laplace 2D stencil kernels -------------------===!
!
! The Fortran counterpart of apps/c/laplace_2d/laplace_kernels.h, written in
! the ops-mlir kernel convention: a kernel is a pure function of its stencil
! neighbours, taking one scalar per stencil point and returning the value
! written at the centre point.
!
! The neighbours arrive in the order the loop's stencil declares them, so for
! S2D_5pt = [0,0, 1,0, -1,0, 0,1, 0,-1] the parameters are the centre and
! then east, west, north, south.
!
! The JIT compiles this module with flang at run time and translates these
! bodies, rather than calling into compiled host code.
!
!===----------------------------------------------------------------------===!

module laplace_kernels

    use OPS_CONSTANTS

    implicit none

contains

    real(kind=8) function set_zero()
        set_zero = 0.0_8
    end function set_zero

    real(kind=8) function warmup()
        warmup = 0.0_8
    end function warmup

    real(kind=8) function copy(a)
        real(kind=8), intent(in) :: a
        copy = a
    end function copy

    real(kind=8) function left_bndcon(idx)
        integer(kind=4), intent(in) :: idx(2)
        left_bndcon = sin(pi * (idx(2) + 1) / (jmax + 1))
    end function left_bndcon

    real(kind=8) function right_bndcon(idx)
        integer(kind=4), intent(in) :: idx(2)
        right_bndcon = sin(pi * (idx(2) + 1) / (jmax + 1)) * exp(-pi)
    end function right_bndcon

    real(kind=8) function apply_stencil(a, b, c, d, e)
        real(kind=8), intent(in) :: a, b, c, d, e
        apply_stencil = 0.25_8 * (b + c + d + e)
    end function apply_stencil

end module laplace_kernels
