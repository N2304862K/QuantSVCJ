#ifndef SVCJ_H
#define SVCJ_H

#include <math.h>
#include <stdlib.h>
#include <string.h>

// Configuration
#define MAX_ITER 100
#define OPT_TOL 1e-5
#define DT 1.0 // Daily
#define MERTON_ITER 20 // For Option Pricing Series

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
double ukf_log_likelihood(double* returns, int n, SVCJParams* params, double* out_spot_vol, double* out_jump_prob);
void optimize_svcj(double* returns, int n, SVCJParams* params, double* out_spot_vol, double* out_jump_prob);

// Pricing
void price_option_chain(double s0, double* strikes, double* expiries, int* types, int n_opts, SVCJParams* params, double spot_vol, double* out_prices);

#endif