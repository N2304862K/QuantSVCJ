#ifndef SVCJ_H
#define SVCJ_H

#include <math.h>
#include <stdlib.h>

// --- CONFIGURATION: ANNUALIZED STATE / DAILY STEP ---
#define DAYS_PER_YEAR 252.0
#define DT (1.0 / DAYS_PER_YEAR) 

// Stability constants
#define MAX_ITER 200
#define JITTER_THRESHOLD 1e-8 
#define MIN_VAR 1e-4 // Floor for annualized variance (1% vol)

typedef struct {
    double mu;          // Annualized Drift
    double kappa;       // Annualized Mean Reversion
    double theta;       // Annualized Long-Run Variance
    double sigma_v;     // Annualized Vol-of-Vol
    double rho;         // Correlation
    double lambda_j;    // Annualized Jump Intensity
    double mu_j;        // Mean Jump Size (on return scale)
    double sigma_j;     // Jump Size Std Dev
} SVCJParams;

// Core Functions
void clean_and_copy(double* src, double* dest, int n); // New safe copy utility
void check_stability(SVCJParams* params);
double run_ukf_qmle(double* returns, int n, SVCJParams* params, double* out_spot_vol, double* out_jump_prob);
void price_option_chain(double s0, double* strikes, double* expiries, int* types, int n_opts, SVCJParams* params, double* out_prices);

#endif