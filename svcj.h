#ifndef SVCJ_H
#define SVCJ_H

#include <math.h>
#include <stdlib.h>

// Configuration Constants
#define MAX_ITER 200
#define TOLERANCE 1e-6
#define JITTER_THRESHOLD 1e-8
#define DT 1.0 // Daily steps implicit

// Structure to hold SVCJ Model Parameters
typedef struct {
    double mu;          // Drift
    double kappa;       // Mean reversion speed
    double theta;       // Long-run variance
    double sigma_v;     // Vol of Vol
    double rho;         // Correlation (Brownian)
    double lambda_j;    // Jump intensity
    double mu_j;        // Mean jump size
    double sigma_j;     // Jump size uncertainty
} SVCJParams;

// Structure for Unscented Kalman Filter State
typedef struct {
    double spot_vol;    // Filtered Spot Volatility
    double jump_prob;   // Instantaneous Jump Probability
    double likelihood;  // Log-Likelihood of this state
} UKFState;

// Core C Functions exposed to Cython
void clean_returns(double* returns, int n);
void check_feller_and_fix(SVCJParams* params);
double run_ukf_qmle(double* returns, int n, SVCJParams* params, double* out_spot_vol, double* out_jump_prob);
void price_option_chain(double s0, double* strikes, double* expiries, int* types, int n_opts, SVCJParams* params, double* out_prices);

#endif