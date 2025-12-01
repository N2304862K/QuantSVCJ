#ifndef SVCJ_H
#define SVCJ_H

#include <math.h>
#include <stdlib.h>
<<<<<<< HEAD
#include <stdio.h>

// --- Config ---
#define DT 1.0              // 1 Day
#define DAYS_PER_YEAR 252.0
#define JITTER 1e-9
#define MAX_OPT_ITER 50     // Optimization steps per asset
#define LEARN_RATE 0.1      // Initial learning step for parameters
=======

// Configuration Constants (tuned for Daily Scale)
#define MAX_ITER 200
#define TOLERANCE 1e-6
#define JITTER_THRESHOLD 1e-9  // Lower jitter for daily returns
#define DT 1.0                 // 1 Day step
>>>>>>> d24ae54 (ok)

// Structure to hold SVCJ Model Parameters
typedef struct {
    double mu;          // Drift (Daily)
    double kappa;       // Mean reversion speed (Daily)
    double theta;       // Long-run variance (Daily)
    double sigma_v;     // Vol of Vol (Daily)
    double rho;         // Correlation
    double lambda_j;    // Jump intensity (Daily prob)
    double mu_j;        // Mean jump size
    double sigma_j;     // Jump size uncertainty
} SVCJParams;

<<<<<<< HEAD
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
=======
// Core C Functions exposed to Cython
// Added 'stride' to handle Matrix columns directly without copying
void clean_returns(double* returns, int n, int stride);
void check_feller_and_fix(SVCJParams* params);
double run_ukf_qmle(double* returns, int n, int stride, SVCJParams* params, double* out_spot_vol, double* out_jump_prob);
void price_option_chain(double s0, double* strikes, double* expiries, int* types, int n_opts, SVCJParams* params, double* out_prices);
>>>>>>> d24ae54 (ok)

#endif