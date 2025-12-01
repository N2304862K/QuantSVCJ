#ifndef SVCJ_H
#define SVCJ_H

#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

// --- Configuration ---
#define MAX_ITER 50       // Optimization iterations
#define TOLERANCE 1e-6
#define JITTER_THRESHOLD 1e-8
#define DT (1.0/252.0)    // Daily steps annualized
#define PENALTY_VAL 1e9   // For failed likelihoods

// --- Data Structures ---
typedef struct {
    double mu;          // Drift
    double kappa;       // Mean reversion speed
    double theta;       // Long-run variance
    double sigma_v;     // Vol of Vol
    double rho;         // Correlation
    double lambda_j;    // Jump intensity
    double mu_j;        // Mean jump size
    double sigma_j;     // Jump size uncertainty
} SVCJParams;

// --- Function Prototypes ---

// Utils
void clean_returns(double* returns, int n);
void check_feller_and_fix(SVCJParams* params);
double estimate_initial_variance(double* returns, int n);

// Core Computation
double run_ukf_qmle(double* returns, int n, SVCJParams* params, double* out_spot_vol, double* out_jump_prob);
void fit_svcj_history(double* returns, int n, SVCJParams* params);

// Option Handling
void calibrate_option_adjustment(double s0, double* strikes, double* expiries, int* types, double* mkt_prices, int n_opts, SVCJParams* params);

#endif