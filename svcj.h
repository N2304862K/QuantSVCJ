#ifndef SVCJ_H
#define SVCJ_H

#include <math.h>
#include <stdlib.h>
#include <string.h>

// --- Configuration ---
#define MAX_ITER 50
#define SUB_STEPS 20        // Integrate diffusion 20 times per daily step for stability
#define DT 1.0              // 1 Day
#define SUB_DT (DT/SUB_STEPS)
#define JIT_THRESH 1e-8

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

// --- Prototypes ---
void clean_returns(double* returns, int n);
void check_constraints(SVCJParams* params);

// Optimization & Filtering
double ukf_log_likelihood(double* returns, int n, SVCJParams* params, double* out_spot_vol, double* out_jump_prob);
void optimize_svcj(double* returns, int n, SVCJParams* params, double* out_spot_vol, double* out_jump_prob);

// Pricing
void price_option_chain(double s0, double* strikes, double* expiries, int* types, int n_opts, SVCJParams* params, double spot_vol, double* out_prices);

#endif