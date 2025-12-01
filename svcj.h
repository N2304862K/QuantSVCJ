#ifndef SVCJ_H
#define SVCJ_H

#include <math.h>
#include <stdlib.h>
#include <float.h>

// Configuration
#define DT 1.0/252.0 // Annualized Time Step
#define MIN_VAR 1e-6
#define MAX_VAR 10.0

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
void clean_returns(double* returns, int n, int stride);
void check_feller_and_fix(SVCJParams* params);

// Returns Negative Log Likelihood
double run_ukf_qmle(double* returns, int n, int stride, SVCJParams* params, double* out_spot_vol, double* out_jump_prob);

// Simple Coordinate Descent to fit parameters
void optimize_svcj(double* returns, int n, int stride, SVCJParams* params);

void price_option_chain(double s0, double* strikes, double* expiries, int* types, int n_opts, SVCJParams* params, double* out_prices);

#endif