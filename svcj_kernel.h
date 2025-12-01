#ifndef SVCJ_KERNEL_H
#define SVCJ_KERNEL_H

#include <math.h>
#include <stdlib.h>
#include <string.h>

// Parameters in DAILY units for stability
typedef struct {
    double kappa;       // Mean reversion 
    double theta;       // Long-run variance (Daily units: e.g. 0.00016 for 20% ann vol)
    double sigma_v;     // Vol of vol
    double rho;         // Correlation
    double lambda_j;    // Jump intensity (Daily prob)
    double mu_j;        // Mean jump size
    double sigma_j;     // Jump size uncertainty
    double v0;          // Initial variance (Daily units)
} SVCJParams;

typedef struct {
    double spot_vol;       // Annualized Volatility output
    double jump_prob;      // Posterior Probability of Jump
    double drift_residue;  // Realized - Expected
    double vt;             // Internal Daily Variance
} FilterState;

typedef struct {
    double strike;
    double price;
    double T;
    int is_call;
} OptionContract;

void optimize_params_history(double* returns, int n_steps, SVCJParams* out_best_params);
void run_ukf_filter(double* returns, int n_steps, SVCJParams* params, FilterState* out_states);
void calibrate_to_options(OptionContract* options, int n_opts, double spot_price, SVCJParams* out_params);

#endif