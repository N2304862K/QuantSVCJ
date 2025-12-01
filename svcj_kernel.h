#ifndef SVCJ_KERNEL_H
#define SVCJ_KERNEL_H

#include <math.h>
#include <stdlib.h>
#include <string.h>

// --- Data Structures ---
typedef struct {
    double kappa;       // Daily Mean Reversion
    double theta;       // Daily Long-Run Variance
    double sigma_v;     // Daily Vol-of-Vol
    double rho;         // Correlation
    double lambda_j;    // Daily Jump Intensity
    double mu_j;        // Mean Jump Size
    double sigma_j;     // Jump Uncertainty
    double v0;          // Current Daily Variance state
} SVCJParams;

typedef struct {
    double spot_vol;       // Daily Vol
    double jump_prob;      // 0.0 - 1.0
    double drift_residue;  
} FilterState;

typedef struct {
    double strike;
    double price;
    double T_years;
    int is_call;
    int valid; // Flag for C-level filtering
} OptionContract;

// --- Prototypes ---

// 1. Data Transformation (New: Raw Price -> Returns)
void calculate_returns_from_prices(double* prices, int n_prices, double* out_returns);

// 2. Option Surface Smoothing (New: Filtering logic)
void preprocess_options(OptionContract* opts, int n_opts, double spot_price);

// 3. Core Physics
void optimize_params_history(double* returns, int n_steps, SVCJParams* out);
void run_ukf_filter(double* returns, int n_steps, SVCJParams* params, FilterState* out_states);
void calibrate_to_options(OptionContract* options, int n_opts, double spot_price, SVCJParams* out_params);

#endif