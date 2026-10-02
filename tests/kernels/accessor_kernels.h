// Accessor-style (ACC<T>) kernels for the e2e tests of reductions, loops over fewer axes and
// registered array constants -- the shapes CloverLeaf uses. They are parsed by the JIT from this
// file and also compiled into the test, which runs them as the reference through the stock OPS.

struct tstate_type {
  double scale;
  double shift;
  int kind;
};

// OPS checks the type name of a registered constant with an overload like this one.
inline int type_error(const tstate_type *, const char *type) { return __builtin_strcmp(type, "tstate_type"); }

extern tstate_type *tstates;
extern int tnum_states;

// X(i, j) = idx-based pattern
void a_init(ACC<double> &x, const int *idx) {
  x(0, 0) = idx[0] * 0.5 + idx[1] * 0.25 + 1.0;
}

// sum, min and max of a dat, one loop
void a_stats(const ACC<double> &x, double *sum, double *lo, double *hi) {
  *sum += x(0, 0);
  if (x(0, 0) < *lo)
    *lo = x(0, 0);
  if (x(0, 0) > *hi)
    *hi = x(0, 0);
}

// the accumulation written out: r = r + e
void a_sum_spelled(const ACC<double> &x, double *sum, double *sq) {
  *sum = *sum + x(0, 0);
  *sq = x(0, 0) * x(0, 0) + *sq;
}

// a reduction with two elements
void a_sum2(const ACC<double> &x, double *s) {
  s[0] += x(0, 0);
  s[1] += x(0, 0) * 2.0;
}

// assigns the reduction: only meaningful for a loop over one point
void a_pick(const ACC<double> &x, double *v) {
  *v = x(0, 0);
}

// a 1-D array along x written from the index, a 1-D array along y likewise
void a_coord_x(ACC<double> &cx, const int *idx) {
  cx(0, 0) = idx[0] * 1.5 - 2.0;
}

void a_coord_y(ACC<double> &cy, const int *idx) {
  cy(0, 0) = idx[1] * 0.75 + 4.0;
}

// reads a 1-D array at a neighbour and writes another one
void a_coord_cell(const ACC<double> &cx, ACC<double> &cell) {
  cell(0, 0) = 0.5 * (cx(0, 0) + cx(1, 0));
}

// a full 2-D dat from the two 1-D arrays
void a_outer(const ACC<double> &cx, const ACC<double> &cy, ACC<double> &o) {
  o(0, 0) = cx(0, 0) * cy(0, 0);
}

// registered array of structs, read in an unrolled loop with a data-dependent branch
void a_states(ACC<double> &out) {
  out(0, 0) = 1.0;
  for (int s = 0; s < tnum_states; s++) {
    if (tstates[s].kind == 1)
      out(0, 0) = out(0, 0) * tstates[s].scale + tstates[s].shift;
  }
}
