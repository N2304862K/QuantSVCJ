#ifndef SVCJ_KERNEL_H
#define SVCJ_KERNEL_H

#include <math.h>
#include <stdlib.h>

// --- Data Structures ---

typedef struct {
    double kappa;       // Mean reversion speed
    double theta;       // Long-run variance
    double sigma_v;     // Vol of vol
    double rho;         // Correlation (price, vol)
    double lambda_j;    // Jump intensity
    double mu_j;        // Mean jump size
    double sigma_j;     // Jump size uncertainty
    double v0;          // Initial volatility
} SVCJParams;

typedef struct {
    double spot_vol;
    double jump_prob;
    double log_likelihood;
    double drift_residue;
} FilterState;

typedef struct {
    double strike;
    double price;
    double T;
    int is_call; // 1 for Call, 0 for Put
} OptionContract;

// --- Prototypes ---

// 1. Core Logic: UKF Filter & QMLE
void run_ukf_filter(double* returns, int n_steps, SVCJParams* params, FilterState* out_states);
double calculate_neg_log_likelihood(double* returns, int n_steps, SVCJParams* params);

// 2. Optimization & Constraints
void enforce_feller(SVCJParams* p);
void optimize_params_history(double* returns, int n_steps, SVCJParams* out_best_params);

// 3. Option Integration (Carr-Madan FFT approach simplified for C)
void calibrate_to_options(OptionContract* options, int n_opts, double spot_price, SVCJParams* out_params);

// 4. Utilities
void denoise_returns(double* returns, int n_steps, double jitter);

#endif