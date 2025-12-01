#ifndef SVCJ_KERNEL_H
#define SVCJ_KERNEL_H

#include <math.h>
#include <stdlib.h>
#include <string.h>

// --- Data Structures (DAILY TERMS) ---
typedef struct {
    double kappa;       // Daily mean reversion speed
    double theta;       // Long-run DAILY variance
    double sigma_v;     // Vol of vol (Daily scale)
    double rho;         // Correlation (-1 to 1)
    double lambda_j;    // Daily jump intensity
    double mu_j;        // Mean jump size
    double sigma_j;     // Jump size uncertainty
    double v0;          // Current DAILY variance
} SVCJParams;

typedef struct {
    double spot_vol;       // Daily Volatility (sqrt(v0))
    double jump_prob;      // Instantaneous probability
    double drift_residue;  // Realized Return - Expected Daily Drift
    double vt;             // Internal variance state
} FilterState;

typedef struct {
    double strike;
    double price;
    double T;      // Time in Years (for Black-Scholes inversion only)
    int is_call;
} OptionContract;

// --- Prototypes ---
void optimize_params_history(double* returns, int n_steps, SVCJParams* out_best_params);
void run_ukf_filter(double* returns, int n_steps, SVCJParams* params, FilterState* out_states);
void calibrate_to_options(OptionContract* options, int n_opts, double spot_price, SVCJParams* out_params);

#endif