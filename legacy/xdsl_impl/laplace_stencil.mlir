builtin.module {
  func.func private @set_zero() -> f64
  func.func private @ops_par_loop_set_zero_0(%0: !stencil.field<[0,4096]x[0,4096]xf64>) {
    stencil.apply() outs (%0 : !stencil.field<[0,4096]x[0,4096]xf64>) {
      %1 = func.call @set_zero() : () -> f64
      stencil.return %1 : f64
    } to <[0, 0], [4096, 4096]>
    func.return
  }
  func.func private @ops_par_loop_set_zero_1(%0: !stencil.field<[0,4096]x[0,4096]xf64>) {
    stencil.apply() outs (%0 : !stencil.field<[0,4096]x[0,4096]xf64>) {
      %1 = func.call @set_zero() : () -> f64
      stencil.return %1 : f64
    } to <[0, 0], [4096, 4096]>
    func.return
  }
  func.func private @ops_par_loop_set_zero_2(%0: !stencil.field<[0,4096]x[0,4096]xf64>) {
    stencil.apply() outs (%0 : !stencil.field<[0,4096]x[0,4096]xf64>) {
      %1 = func.call @set_zero() : () -> f64
      stencil.return %1 : f64
    } to <[0, 0], [4096, 1]>
    func.return
  }
  func.func private @ops_par_loop_set_zero_3(%0: !stencil.field<[0,4096]x[0,4096]xf64>) {
    stencil.apply() outs (%0 : !stencil.field<[0,4096]x[0,4096]xf64>) {
      %1 = func.call @set_zero() : () -> f64
      stencil.return %1 : f64
    } to <[0, 4095], [4096, 4096]>
    func.return
  }
  func.func private @left_bndcon(index, index) -> f64
  func.func private @ops_par_loop_left_bndcon_4(%0: !stencil.field<[0,4096]x[0,4096]xf64>) {
    stencil.apply() outs (%0 : !stencil.field<[0,4096]x[0,4096]xf64>) {
      %1 = stencil.index 0 <[-1, 0]>
      %2 = stencil.index 1 <[0, -1]>
      %3 = func.call @left_bndcon(%1, %2) : (index, index) -> f64
      stencil.return %3 : f64
    } to <[0, 0], [1, 4096]>
    func.return
  }
  func.func private @right_bndcon(index, index) -> f64
  func.func private @ops_par_loop_right_bndcon_5(%0: !stencil.field<[0,4096]x[0,4096]xf64>) {
    stencil.apply() outs (%0 : !stencil.field<[0,4096]x[0,4096]xf64>) {
      %1 = stencil.index 0 <[-1, 0]>
      %2 = stencil.index 1 <[0, -1]>
      %3 = func.call @right_bndcon(%1, %2) : (index, index) -> f64
      stencil.return %3 : f64
    } to <[4095, 0], [4096, 4096]>
    func.return
  }
}
