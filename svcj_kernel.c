#include "svcj_kernel.h"
#include <stdio.h>
#include <float.h>

#define PI 3.14159265358979323846
#define SQRT_2PI 2.50662827463
#define INF_LOSS 1e15

// --- 1. Data Transformation Logic ---
void calculate_returns_from_prices(double* prices, int n_prices, double* out_returns) {
    // Calculates Log Returns: ln(P_t / P_{t-1})
    // Handles zeros or negatives by skipping/filling 0
    for (int i = 1; i < n_prices; i++) {
        if (prices[i] > 0 && prices[i-1] > 0) {
            out_returns[i-1] = log(prices[i] / prices[i-1]);
        } else {
            out_returns[i-1] = 0.0;
        }
    }
}

// --- 2. Option Surface Smoothing (The "Intelligent Layer") ---
void preprocess_options(OptionContract* opts, int n_opts, double spot) {
    for(int i=0; i<n_opts; i++) {
        opts[i].valid = 1;
        
        // Filter 1: Deep OTM/ITM (Noise Removal)
        // We only trust options within 80% to 120% of spot
        double moneyness = opts[i].strike / spot;
        if (moneyness < 0.80 || moneyness > 1.20) {
            opts[i].valid = 0;
            continue;
        }

        // Filter 2: Price Granularity (Remove penny options)
        if (opts[i].price < 0.05) {
            opts[i].valid = 0;
            continue;
        }
        
        // Filter 3: Time Horizon (Remove 0DTE noise and LEAPS > 1 year)
        if (opts[i].T_years < (1.0/52.0) || opts[i].T_years > 1.0) {
            opts[i].valid = 0;
            continue;
        }
    }
}

// --- 3. Physics & Constraints ---
double density_norm(double x, double mean, double var) {
    if (var < 1e-9) var = 1e-9;
    return (1.0 / (SQRT_2PI * sqrt(var))) * exp(-0.5 * (x - mean)*(x - mean) / var);
}

int check_constraints(SVCJParams* p) {
    // Physical "Directionality" Checks for Daily Variance
    if (p->theta < 1e-7 || p->theta > 0.005) return 0; // Daily Variance bounds
    if (p->v0 < 1e-7 || p->v0 > 0.01) return 0;
    if (p->kappa < 0.001 || p->kappa > 1.0) return 0; // Daily Mean Reversion
    if (p->sigma_v < 1e-4 || p->sigma_v > 0.5) return 0;
    if (p->rho < -0.99 || p->rho > 0.99) return 0;
    
    // Feller Violation Check (Soft)
    double lhs = 2.0 * p->kappa * p->theta;
    double rhs = p->sigma_v * p->sigma_v;
    if (lhs < rhs * 0.5) return 0; // Allow slight violation, reject extreme

    return 1;
}

// --- 4. Optimization Core ---
double calculate_nll(double* returns, int n_steps, SVCJParams* p) {
    if (!check_constraints(p)) return INF_LOSS;
    double nll = 0.0;
    double vt = p->v0;
    double jump_var_contrib = p->lambda_j * (p->mu_j*p->mu_j + p->sigma_j*p->sigma_j);

    for (int i = 0; i < n_steps; i++) {
        if (vt < 1e-8) vt = 1e-8;
        if (vt > 0.01) vt = 0.01;

        double drift = p->kappa * (p->theta - vt);
        double vt_pred = vt + drift;
        double total_var = vt_pred + jump_var_contrib;
        if (total_var < 1e-8) total_var = 1e-8;

        double lik = density_norm(returns[i], 0.0, total_var);
        if (lik < 1e-15) lik = 1e-15;
        nll -= log(lik);

        double shock = (returns[i]*returns[i]) - total_var;
        vt = vt_pred + (p->sigma_v * 0.15) * shock; // GARCH-update proxy
    }
    return nll;
}

void optimize_params_history(double* returns, int n_steps, SVCJParams* out) {
    // 1. Initialize from Realized Var
    double sum_sq = 0.0;
    for(int i=0; i<n_steps; i++) sum_sq += returns[i]*returns[i];
    double rv = sum_sq / n_steps;
    
    out->theta = rv;
    out->v0 = rv;
    out->lambda_j = 0.05; out->mu_j = -0.01; out->sigma_j = sqrt(rv)*3.0;

    // 2. Grid Search
    double kappas[] = {0.02, 0.05, 0.10};
    double rhos[] = {-0.7, -0.4, -0.1}; // Equity bias negative
    double sigmas[] = {0.001, 0.005, 0.01}; 

    double best_nll = INF_LOSS;
    SVCJParams best_p = *out;
    SVCJParams candidate = *out;

    for(int k=0; k<3; k++) {
        for(int r=0; r<3; r++) {
            for(int s=0; s<3; s++) {
                candidate.kappa = kappas[k];
                candidate.rho = rhos[r];
                candidate.sigma_v = sigmas[s];
                
                double nll = calculate_nll(returns, n_steps, &candidate);
                if (nll < best_nll) {
                    best_nll = nll;
                    best_p = candidate;
                }
            }
        }
    }
    *out = best_p;
}

// --- 5. Option Calibration (With Smoothing) ---
void calibrate_to_options(OptionContract* options, int n_opts, double spot, SVCJParams* out) {
    preprocess_options(options, n_opts, spot); // Filter noise first

    double sum_var = 0.0;
    int count = 0;
    double skew_accum = 0.0;

    for(int i=0; i<n_opts; i++) {
        if(options[i].valid == 0) continue; // Skip filtered

        double moneyness = options[i].strike / spot;
        double t_sqrt = sqrt(options[i].T_years);
        
        // ATM Vol Extraction
        if (fabs(moneyness - 1.0) < 0.05) {
            double vol_ann = (options[i].price / spot * 2.50) / t_sqrt;
            double var_daily = (vol_ann * vol_ann) / 252.0;
            sum_var += var_daily;
            count++;
        }
        // Skew
        if (moneyness < 0.95 && options[i].is_call == 0) skew_accum -= 1.0;
    }

    if (count > 0) {
        double implied_theta = sum_var / count;
        // Weight: 70% Option Implied, 30% Historical
        out->theta = (0.7 * implied_theta) + (0.3 * out->theta);
        out->v0 = implied_theta; 
    }
    
    if (skew_accum < -5.0) out->rho = -0.85; // Heavy Put Skew
}

// --- 6. UKF Runtime ---
void run_ukf_filter(double* returns, int n_steps, SVCJParams* p, FilterState* out_states) {
    double vt = p->v0;
    for (int i = 0; i < n_steps; i++) {
        if (vt < 1e-8) vt = 1e-8;

        // Predict
        double drift = p->kappa * (p->theta - vt);
        double vt_pred = vt + drift;
        double total_var = vt_pred + (p->lambda_j * (p->mu_j*p->mu_j + p->sigma_j*p->sigma_j));
        
        // Update
        double innovation = returns[i];
        double gain = (p->sigma_v * p->rho) / (total_var + 1e-9);
        
        vt = vt_pred + gain * (innovation*innovation - total_var);
        
        if (vt < 1e-8) vt = 1e-8;
        if (vt > 0.02) vt = 0.02;

        // Jump Prob
        double z = innovation / sqrt(vt_pred);
        double jump_p = (fabs(z) > 3.5) ? (1.0 - exp(-(fabs(z)-3.5))) : 0.0;

        out_states[i].spot_vol = sqrt(vt);
        out_states[i].jump_prob = jump_p;
        out_states[i].drift_residue = innovation;
    }
}