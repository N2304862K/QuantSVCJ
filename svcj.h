#ifndef SVCJ_H
#define SVCJ_H

#include <math.h>
#include <stdlib.h>

// Configuration Constants (tuned for Daily Scale)
#define MAX_ITER 200
#define TOLERANCE 1e-6
#define JITTER_THRESHOLD 1e-9  // Lower jitter for daily returns
#define DT 1.0                 // 1 Day step

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

// Core C Functions exposed to Cython
// Added 'stride' to handle Matrix columns directly without copying
void clean_returns(double* returns, int n, int stride);
void check_feller_and_fix(SVCJParams* params);
double run_ukf_qmle(double* returns, int n, int stride, SVCJParams* params, double* out_spot_vol, double* out_jump_prob);
void price_option_chain(double s0, double* strikes, double* expiries, int* types, int n_opts, SVCJParams* params, double* out_prices);

#endif