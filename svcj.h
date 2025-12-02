#ifndef SVCJ_H
#define SVCJ_H

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

// Configuration
#define DT (1.0/252.0)     // Time step (Daily)
#define NM_ITER 250        // Increased Iterations for MAP convergence
#define SQRT_2PI 2.50662827463

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
void check_constraints(SVCJParams* params);

// Optimization & Filtering
void estimate_initial_params(double* returns, int n, SVCJParams* p);
double ukf_log_likelihood(double* returns, int n, SVCJParams* params, double* out_spot_vol, double* out_jump_prob);
void optimize_svcj(double* returns, int n, SVCJParams* params, double* out_spot_vol, double* out_jump_prob);

// Pricing
void price_option_chain(double s0, double* strikes, double* expiries, int* types, int n_opts, SVCJParams* params, double spot_vol, double* out_prices);

#endif