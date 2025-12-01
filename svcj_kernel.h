#ifndef SVCJ_KERNEL_H
#define SVCJ_KERNEL_H

#include <math.h>
#include <stdlib.h>
#include <string.h>

// --- Structures ---
typedef struct {
    double kappa;
    double theta;
    double sigma_v;
    double rho;
    double lambda_j;
    double mu_j;
    double sigma_j;
    double v0;
} SVCJParams;

typedef struct {
    double strike;
    double price;
    double T_days; // Raw days to expiry
    int is_call;
} RawOption;

typedef struct {
    double spot_vol;
    double jump_prob;
    double drift_residue;
} FilterResult;

// --- Core API ---
// Accepts RAW Prices, computes returns internally, processes options, outputs vectors
void full_svcj_pipeline(
    double* raw_prices, 
    int n_price_steps,
    RawOption* raw_options, 
    int n_opts,
    SVCJParams* out_params, 
    FilterResult* out_states
);

// Rolling analysis on raw price matrix
void batch_process_matrix(
    double* price_matrix, 
    int n_assets, 
    int n_time, 
    double* out_spot_vols, 
    double* out_jump_probs
);

#endif