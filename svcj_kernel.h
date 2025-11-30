#ifndef SVCJ_KERNEL_H
#define SVCJ_KERNEL_H

#include <math.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    double kappa;       // Mean reversion (> 0)
    double theta;       // Long-run variance (> 0)
    double sigma_v;     // Vol of vol (> 0)
    double rho;         // Correlation (-1 to 1)
    double lambda_j;    // Jump intensity (0 to 1)
    double mu_j;        // Mean jump size
    double sigma_j;     // Jump size uncertainty
    double v0;          // Initial variance state
} SVCJParams;

typedef struct {
    double spot_vol;       // Annualized Volatility
    double jump_prob;      // Instantaneous probability (0-1)
    double drift_residue;  // Realized Return - Expected Drift
    double vt;             // Internal variance state
} FilterState;

typedef struct {
    double strike;
    double price;
    double T;
    int is_call;
} OptionContract;

// Core Prototypes
void optimize_params_history(double* returns, int n_steps, SVCJParams* out_best_params);
void run_ukf_filter(double* returns, int n_steps, SVCJParams* params, FilterState* out_states);
void calibrate_to_options(OptionContract* options, int n_opts, double spot_price, SVCJParams* out_params);

#endif