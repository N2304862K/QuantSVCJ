#ifndef SVCJ_H
#define SVCJ_H

#include <math.h>
#include <stdlib.h>
#include <stdio.h>

// --- Config ---
#define DT 1.0              // 1 Day
#define DAYS_PER_YEAR 252.0
#define JITTER 1e-9
#define MAX_OPT_ITER 50     // Optimization steps per asset
#define LEARN_RATE 0.1      // Initial learning step for parameters

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

// --- Core API ---
void clean_returns(double* returns, int n, int stride);
void constrain_params(SVCJParams* p);

// The Core Likelihood Function
double run_ukf_likelihood(double* returns, int n, int stride, SVCJParams* p, double* out_spot, double* out_jump);

// Calibration Functions
void calibrate_to_history(double* returns, int n, int stride, SVCJParams* p);
void calibrate_to_options(double s0, double* strikes, double* expiries, int* types, double* mkt_prices, int n_opts, SVCJParams* p);

// Pricing
void price_option_chain(double s0, double* strikes, double* expiries, int* types, int n_opts, SVCJParams* p, double* out_prices);

#endif