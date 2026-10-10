// Optional end-of-run diagnostics for the Taylor-Green vortex app.
//
//   TGV_DIAG=1        print mean kinetic energy, density range, ... to stdout
//   TGV_DUMP=<prefix> write the interior of each field as raw binary
//                     (<prefix>_<field>.bin, x fastest, sizeof(real_t) bytes
//                     per value) plus <prefix>.meta
//
// real_t is the solver precision (constants.h). Accumulators are `double` on
// purpose and are named diag_*: to_single_precision.py leaves this file
// untouched, so the single-precision build reports in double.
#ifndef TGV_DIAGNOSTICS_H
#define TGV_DIAGNOSTICS_H

#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <vector>

static void tgv_fetch(ops_dat dat, std::vector<real_t> &out) {
  out.resize((size_t)block0np0 * block0np1 * block0np2);
  // Flushes queued loops and copies the dat back from the device first.
  ops_dat_fetch_data(dat, 0, (char *)out.data());
}

static void tgv_diagnostics(ops_dat rho_dat, ops_dat rhou0_dat,
                            ops_dat rhou1_dat, ops_dat rhou2_dat,
                            ops_dat rhoE_dat) {
  const char *dump = getenv("TGV_DUMP");
  if (!getenv("TGV_DIAG") && !dump)
    return;

  std::vector<real_t> r, u0, u1, u2, e;
  tgv_fetch(rho_dat, r);
  tgv_fetch(rhou0_dat, u0);
  tgv_fetch(rhou1_dat, u1);
  tgv_fetch(rhou2_dat, u2);
  tgv_fetch(rhoE_dat, e);

  const size_t n = r.size();
  double diag_ke = 0, diag_rho = 0, diag_e = 0;
  double diag_rmin = r[0], diag_rmax = r[0];
  size_t diag_bad = 0;
  for (size_t i = 0; i < n; ++i) {
    double diag_r = r[i], diag_a = u0[i], diag_b = u1[i], diag_c = u2[i];
    if (!(diag_r == diag_r) || !(e[i] == e[i]))
      ++diag_bad;
    diag_ke += 0.5 * (diag_a * diag_a + diag_b * diag_b + diag_c * diag_c) /
               diag_r;
    diag_rho += diag_r;
    diag_e += e[i];
    if (diag_r < diag_rmin) diag_rmin = diag_r;
    if (diag_r > diag_rmax) diag_rmax = diag_r;
  }
  printf("TGV diag: sizeof(real_t)=%d iter=%d t=%.8f KE=%.12e mean_rho=%.12e "
         "mean_rhoE=%.12e rho_min=%.12e rho_max=%.12e nan=%zu\n",
         (int)sizeof(real_t), iter, simulation_time, diag_ke / n, diag_rho / n,
         diag_e / n, diag_rmin, diag_rmax, diag_bad);

  if (dump) {
    const std::string prefix = dump;
    struct Field { const char *name; std::vector<real_t> *v; };
    for (Field f : {Field{"rho", &r}, Field{"rhou0", &u0}, Field{"rhou1", &u1},
                    Field{"rhou2", &u2}, Field{"rhoE", &e}}) {
      FILE *fp = fopen((prefix + "_" + f.name + ".bin").c_str(), "wb");
      if (!fp) continue;
      fwrite(f.v->data(), sizeof(real_t), f.v->size(), fp);
      fclose(fp);
    }
    FILE *meta = fopen((prefix + ".meta").c_str(), "w");
    if (meta) {
      fprintf(meta, "n=%d\nsizeof=%d\niter=%d\nt=%.12e\n", block0np0,
              (int)sizeof(real_t), iter, simulation_time);
      fclose(meta);
    }
  }
}

#endif
