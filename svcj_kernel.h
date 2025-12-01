#ifndef SVCJ_KERNEL_H
#define SVCJ_KERNEL_H

#include <math.h>
#include <stdlib.h>
#include <string.h>

// All variance parameters are in DAILY (per-step) units.
typedef struct {
    double alpha;       // Constant drift component (Daily)
    double kappa;       // Mean reversion speed
    double theta;       // Long-run variance (Daily units)
    double sigma_v;     // Vol of vol
    double rho;         // Correlation
    double lambda_j;    // Jump intensity (Daily probability)
    double mu_j;        // Mean jump size
    double sigma_j;     // Jump size uncertainty
    double v0;          // Initial variance (Daily units)
} SVCJParams;

typedef struct {
    double spot_vol;       // Daily Volatility (sqrt(vt))
    double jump_prob;      // Instantaneous probability
    double drift_residue;  // Realized - Forecast
    double drift_forecast; // Expected return for next step E[r_{t+1}]
    double vt;             // Internal Daily Variance
} FilterState;

typedef struct {
    double strike;
    double price;
    double T; // Years (used for Black-Scholes inversion only)
    int is_call;
} OptionContract;

void optimize_params_history(double* returns, int n_steps, SVCJParams* out_best_params);
void run_ukf_filter(double* returns, int n_steps, SVCJParams* params, FilterState* out_states);
void calibrate_to_options(OptionContract* options, int n_opts, double spot_price, SVCJParams* out_params);

#endif