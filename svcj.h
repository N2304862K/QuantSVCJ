#ifndef SVCJ_H
#define SVCJ_H

#include <math.h>
#include <complex.h>
#include <stdlib.h>

// --- Data Structures ---

typedef struct {
    double kappa;       // Mean reversion speed
    double theta;       // Long-term variance
    double sigma_v;     // Vol of Vol
    double rho;         // Correlation
    double lambda_j;    // Jump intensity
    double mu_j;        // Jump mean
    double sigma_j;     // Jump volatility
    double v0;          // Initial variance
} ModelParams;

typedef struct {
    double spot_vol;            // Extracted latent volatility
    double jump_prob;           // Instantaneous jump probability
    double log_likelihood;      // QMLE Metric
} FilterOutput;

// --- Function Prototypes ---

// Core UKF-QMLE Algorithm
void run_ukf_qmle(double* log_returns, int T, ModelParams* params, FilterOutput* out_states);

// Carr-Madan Option Pricing (Fourier Transform)
double carr_madan_price(double S0, double K, double T, double r, double q, ModelParams* p, int is_call);

// Helper: Feller Condition Enforcer
double check_feller_condition(double kappa, double theta, double sigma_v);

// Helper: Denoise 0.0 returns
void denoise_data(double* data, int n);

#endif