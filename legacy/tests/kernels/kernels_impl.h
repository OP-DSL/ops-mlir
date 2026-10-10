// Kernels shared by the f32/f64 e2e tests. Written against `real_t` and
// RLIT() (a literal-suffix macro) which the including header defines.
//
// Arg order seen by a kernel: one value per stencil point of each read
// dat (in arg order), then read-only globals, then `const int *idx`; the
// result(s) are the return value or an out-struct, one field per written
// dat (in arg order).

real_t k_zero() { return RLIT(0.0); }

real_t k_copy(real_t a) { return a; }

real_t k_init(const int *idx) {
  return idx[0] * RLIT(0.5) + idx[1] * RLIT(0.25) + RLIT(1.0);
}

real_t k_scale2(real_t b) { return RLIT(2.0) * b; }

real_t k_plus1(real_t a) { return a + RLIT(1.0); }

real_t k_square(real_t c) { return c * c; }

real_t k_add(real_t a, real_t b) { return a + b; }

// 5-point average of the 4 neighbours; `a` (centre) is unused.
real_t k_lap5(real_t a, real_t b, real_t c, real_t d, real_t e) {
  return RLIT(0.25) * (b + c + d + e);
}

real_t k_gscale(real_t a, real_t g) { return a * g; }

struct k_two_result {
  real_t o0;
  real_t o1;
};

void k_two(real_t a, k_two_result *out) {
  out->o0 = a + RLIT(1.0);
  out->o1 = a * RLIT(2.0);
}
