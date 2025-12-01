#ifndef SVCJ_KERNEL_H
#define SVCJ_KERNEL_H

#include <math.h>
#include <stdlib.h>
#include <string.h>

// --- Data Structures ---
typedef struct {
    double kappa;       // Mean Reversion (Daily)
    double theta;       // Long-run Variance (Daily Scale ~1e-4)
    double sigma_v;     // Vol-of-Vol (Daily)
    double rho;         // Correlation
    double lambda_j;    // Jump Intensity
    double mu_j;        // Mean Jump Size
    double sigma_j;     // Jump Uncertainty
    double v0;          // Current Variance State
} SVCJParams;

typedef struct {
    double spot_vol;       // Sqrt(vt)
    double jump_prob;      // 0.0 to 1.0
    double drift_residue;  // Innovation - Drift
} FilterState;

typedef struct {
    double strike;
    double price;
    double T_years;
    int is_call;
    int valid; 
} OptionContract;

// --- Prototypes ---
void calculate_returns_from_prices(double* prices, int n_prices, double* out_returns);
void optimize_params_history(double* returns, int n_steps, SVCJParams* out);
void calibrate_to_options(OptionContract* options, int n_opts, double spot_price, SVCJParams* out_params);
void run_ukf_filter(double* returns, int n_steps, SVCJParams* params, FilterState* out_states);

#endif