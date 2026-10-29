!===- laplace2d.F90 - 2D Laplace solver on the OPS-MLIR Fortran frontend -===!
!
! Part of OPS-MLIR Project. Distributed under the MIT License.
!
! The Fortran counterpart of apps/c/laplace_2d/laplace_2d.cpp, and the same
! problem as the OPS Fortran laplace2d tutorial (apps/fortran/
! laplace2d_tutorial/step7 in the OPS repository).
!
! Relative to the stock OPS Fortran application there is no translator step
! and only one substantive change: `use OPS_Fortran_Reference` becomes
! `use OPS_MLIR_Fortran_Reference`, whose ops_par_loop captures loops for the
! JIT instead of running them. Iteration ranges stay in OPS Fortran's
! one-based inclusive form; the capture layer converts them.
!
!               ^
!               |        top
!               |    000000000000   r
!               | l  000000000000   i
!   y-direction | e  000000000000   g
!    (index-j)  | f  000000000000   h
!               | t  000000000000   t
!               |       bottom
!               o------------------>
!                      x-direction
!                       (index-i)
!
!===----------------------------------------------------------------------===!

program laplace

    use OPS_MLIR_Fortran_Reference
    use OPS_CONSTANTS
    use laplace_kernels

    use, intrinsic :: ISO_C_BINDING

    implicit none

    integer, parameter :: iter_max = 10000
    integer :: iter

    real(kind=8), parameter :: tol = 1.0e-6_8
    real(kind=8) :: error

    type(ops_block)     :: grid2D
    type(ops_dat)       :: d_A, d_Anew
    type(ops_reduction) :: h_err

    ! Stencils: a 1-point and a 5-point. The 5-point ordering fixes which
    ! neighbour each apply_stencil parameter receives.
    integer(kind=4)   :: s2D_00(2) = [0,0]
    type(ops_stencil) :: S2D_0pt
    integer(kind=4)   :: s2D_05(10) = [0,0, 1,0, -1,0, 0,1, 0,-1]
    type(ops_stencil) :: S2D_5pt

    integer(kind=4) :: d_p(2) = [1,1]
    integer(kind=4) :: d_m(2) = [-1,-1]
    integer(kind=4) :: size(2)
    integer(kind=4) :: base(2) = [1,1]

    ! Null array used only to declare the ops_dats; OPS allocates.
    real(kind=8), dimension(:), allocatable :: temp

    ! Verification
    real(kind=8), dimension(:), pointer :: readback
    real(kind=8) :: max_bnd_error, expected
    integer :: j, nx, ny

    real(kind=c_double) :: startTime = 0
    real(kind=c_double) :: endTime = 0

    integer :: bottom_range(4), top_range(4), left_range(4), right_range(4)
    integer :: interior_range(4), full_range(4)

    ! The kernel sources sit next to this file, found the same way the C++
    ! app finds its kernel header.
    character(len=*), parameter :: this_file = __FILE__
    character(len=*), parameter :: src_dir = &
        this_file(1:index(this_file, '/', back=.true.) - 1)

    ! Register the kernel module with the JIT, which translates its bodies
    ! with flang. laplace_kernels uses OPS_CONSTANTS, so that module is
    ! registered as a dependency and compiled first.
    call ops_mlir_set_kernel_source_file(src_dir // "/laplace_kernels.F90")
    call ops_mlir_add_kernel_dependency(src_dir // "/laplace_constants.F90")

    imax = 4094 * 2
    jmax = 4094 * 2

    size(1) = imax
    size(2) = jmax

    ! One-based and inclusive at both ends, as OPS Fortran expects.
    full_range     = [0, imax+1,      0, jmax+1]
    bottom_range   = [0, imax+1,      0, 0]
    top_range      = [0, imax+1,      jmax+1, jmax+1]
    left_range     = [0, 0,           0, jmax+1]
    right_range    = [imax+1, imax+1, 0, jmax+1]
    interior_range = [1, imax,        1, jmax]

    !-----------------------OPS Initialization------------------------
    call ops_init(1)

    !-----------------------OPS Declarations--------------------------
    call ops_decl_block(2, grid2D, "grid2D")

    call ops_decl_stencil(2, 1, s2D_00, S2D_0pt, "0pt_stencil")
    call ops_decl_stencil(2, 5, s2D_05, S2D_5pt, "5pt_stencil")

    call ops_decl_dat(grid2D, 1, size, base, d_m, d_p, temp, d_A, "real(kind=8)", "A")
    call ops_decl_dat(grid2D, 1, size, base, d_m, d_p, temp, d_Anew, "real(kind=8)", "Anew")

    ! These reach the JIT, which bakes their values into the kernel bodies it
    ! translates -- left_bndcon/right_bndcon read jmax directly. `pi` needs
    ! no help: it is a Fortran `parameter`, so flang folds its value itself.
    call ops_decl_const("imax", 1, "integer(kind=4)", imax)
    call ops_decl_const("jmax", 1, "integer(kind=4)", jmax)
    call ops_decl_const("pi", 1, "real(kind=8)", pi)

    error = 1.0_8
    call ops_decl_reduction_handle(8, h_err, "real(kind=8)", "err")

    call ops_partition("")

    call ops_timers(startTime)

    !-----------------------Initial and boundary conditions-----------
    call ops_par_loop(set_zero, "set_zero", grid2D, 2, full_range, &
                    & ops_arg_dat(d_A, 1, S2D_0pt, "real(kind=8)", OPS_WRITE))

    call ops_par_loop(set_zero, "set_zero", grid2D, 2, full_range, &
                    & ops_arg_dat(d_Anew, 1, S2D_0pt, "real(kind=8)", OPS_WRITE))

    call ops_par_loop(set_zero, "set_zero", grid2D, 2, bottom_range, &
                    & ops_arg_dat(d_A, 1, S2D_0pt, "real(kind=8)", OPS_WRITE))

    call ops_par_loop(set_zero, "set_zero", grid2D, 2, top_range, &
                    & ops_arg_dat(d_A, 1, S2D_0pt, "real(kind=8)", OPS_WRITE))

    call ops_par_loop(left_bndcon, "left_bndcon", grid2D, 2, left_range, &
                    & ops_arg_dat(d_A, 1, S2D_0pt, "real(kind=8)", OPS_WRITE), &
                    & ops_arg_idx())

    call ops_par_loop(right_bndcon, "right_bndcon", grid2D, 2, right_range, &
                    & ops_arg_dat(d_A, 1, S2D_0pt, "real(kind=8)", OPS_WRITE), &
                    & ops_arg_idx())

    call ops_par_loop(set_zero, "set_zero", grid2D, 2, bottom_range, &
                    & ops_arg_dat(d_Anew, 1, S2D_0pt, "real(kind=8)", OPS_WRITE))

    call ops_par_loop(set_zero, "set_zero", grid2D, 2, top_range, &
                    & ops_arg_dat(d_Anew, 1, S2D_0pt, "real(kind=8)", OPS_WRITE))

    call ops_par_loop(left_bndcon, "left_bndcon", grid2D, 2, left_range, &
                    & ops_arg_dat(d_Anew, 1, S2D_0pt, "real(kind=8)", OPS_WRITE), &
                    & ops_arg_idx())

    call ops_par_loop(right_bndcon, "right_bndcon", grid2D, 2, right_range, &
                    & ops_arg_dat(d_Anew, 1, S2D_0pt, "real(kind=8)", OPS_WRITE), &
                    & ops_arg_idx())

    call ops_mlir_compile_and_execute()

    !-----------------------Jacobi iteration--------------------------
    iter = 0
    do while (iter < iter_max)

        call ops_par_loop(apply_stencil, "apply_stencil", grid2D, 2, interior_range, &
                        & ops_arg_dat(d_A,    1, S2D_5pt, "real(kind=8)", OPS_READ), &
                        & ops_arg_dat(d_Anew, 1, S2D_0pt, "real(kind=8)", OPS_WRITE), &
                        & ops_arg_reduce(h_err, 1, "real(kind=8)", OPS_MAX))

        call ops_par_loop(copy, "copy", grid2D, 2, interior_range, &
                        & ops_arg_dat(d_A,    1, S2D_0pt, "real(kind=8)", OPS_WRITE), &
                        & ops_arg_dat(d_Anew, 1, S2D_0pt, "real(kind=8)", OPS_READ))

        call ops_mlir_compile_and_execute()
        iter = iter + 1

    end do

    call ops_timers(endTime)
    call ops_timing_output()

    !-----------------------Verification------------------------------
    ! The left boundary is set analytically by left_bndcon and never
    ! touched again, so it pins down the FIR-imported kernel body exactly:
    ! the folded `pi`, the baked `jmax`, and the one-based idx(2) subscript
    ! all have to be right for these to match.
    call ops_mlir_sync_all_host_buffers()

    nx = d_A%dataPtr%size(1)
    ny = d_A%dataPtr%size(2)
    call c_f_pointer(d_A%dataPtr%data, readback, [int(nx, 8) * int(ny, 8)])

    ! The dat is stored with the first dimension contiguous, so the element
    ! at padded (x, y) sits at offset y*nx + x. d_m = -1, so padded x = 0 is
    ! OPS index -1 -- the column left_bndcon writes -- and padded y = j is
    ! OPS index j-1, for which the kernel computes
    ! sin(pi * ((j-1) + 1) / (jmax + 1)).
    max_bnd_error = 0.0_8
    do j = 0, ny - 1
        expected = sin(pi * j / (jmax + 1))
        max_bnd_error = max(max_bnd_error, &
            abs(readback(int(j, 8) * int(nx, 8) + 1) - expected))
    end do
    write(*,'(a,e18.8)') 'Max left-boundary error vs analytic: ', max_bnd_error

    if (max_bnd_error < 1.0e-12_8) then
        write(*,'(a)') 'Boundary check PASSED'
    else
        write(*,'(a)') 'Boundary check FAILED'
    end if

    write(*,'(a,i5,a,i5,a)') 'Jacobi relaxation Calculation:', imax+2, ' x', jmax+2, ' mesh'
    write(*,'(a,i5,a)') 'Completed ', iter, ' iterations'
    write(*,'(a,f16.7,a)') 'Total wall time ', endTime - startTime, ' seconds'

    call ops_mlir_exit()

end program laplace
