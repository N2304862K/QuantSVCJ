#ifndef SVCJ_KERNEL_H
#define SVCJ_KERNEL_H

#include <math.h>
#include <stdlib.h>
#include <string.h>

// --- Data Structures ---

typedef struct {
    double kappa;       // Mean reversion speed
    double theta;       // Long-run variance (annualized)
    double sigma_v;     // Vol of vol
    double rho;         // Correlation
    double lambda_j;    // Jump intensity
    double mu_j;        // Mean jump size
    double sigma_j;     // Jump size uncertainty
    double v0;          // Initial variance (annualized)
} SVCJParams;

typedef struct {
    double spot_vol;       // Annualized Volatility (sqrt(vt))
    double jump_prob;      // Instantaneous probability
    double drift_residue;  // Realized - Expected
    double vt;             // Internal variance state
} FilterState;

typedef struct {
    double strike;
    double price;
    double T;
    int is_call;
} OptionContract;

// --- Prototypes ---

// 1. Core Logic: UKF & Likelihood
void run_ukf_filter(double* returns, int n_steps, SVCJParams* params, FilterState* out_states);
double calculate_neg_log_likelihood(double* returns, int n_steps, SVCJParams* params);

// 2. Optimization (Coordinate Descent)
void optimize_params_history(double* returns, int n_steps, SVCJParams* out_best_params);

// 3. Option Integration
void calibrate_to_options(OptionContract* options, int n_opts, double spot_price, SVCJParams* out_params);

// 4. Utilities
void enforce_feller(SVCJParams* p);

#endif