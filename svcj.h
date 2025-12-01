#ifndef SVCJ_H
#define SVCJ_H

#include <math.h>
#include <stdlib.h>
#include <stdio.h>

// Configuration
#define MAX_ITER 50       // Optimization iterations
#define TOLERANCE 1e-4    // Convergence threshold
#define JITTER_THRESHOLD 1e-8
#define MIN_VAR 1e-6      // Hard floor for variance
#define DT 1.0            // Daily data implicit

typedef struct {
    double mu;
    double kappa;
    double theta;
    double sigma_v;
    double rho;
    double lambda_j;
    double mu_j;
    double sigma_j;
} SVCJParams;

// Core Functions
void clean_returns(double* returns, int n);
void check_feller_and_fix(SVCJParams* params);
double run_ukf_qmle(double* returns, int n, SVCJParams* params, double* out_spot_vol, double* out_jump_prob);
void optimize_svcj_params(double* returns, int n, SVCJParams* params); // NEW: The Solver
void price_option_chain(double s0, double* strikes, double* expiries, int* types, int n_opts, SVCJParams* params, double* out_prices);

#endif