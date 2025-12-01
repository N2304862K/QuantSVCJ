#include "svcj_kernel.h"
#include <stdio.h>
#include <float.h>

#define PI 3.14159265358979323846
#define SQRT_2PI 2.50662827463
#define INF_LOSS 1e20

// --- Utilities ---
int compare_doubles(const void* a, const void* b) {
    double arg1 = *(const double*)a;
    double arg2 = *(const double*)b;
    if (arg1 < arg2) return -1;
    if (arg1 > arg2) return 1;
    return 0;
}

double density_norm(double x, double mean, double var) {
    if (var < 1e-9) var = 1e-9;
    return (1.0 / (SQRT_2PI * sqrt(var))) * exp(-0.5 * (x - mean)*(x - mean) / var);
}

void calculate_returns_from_prices(double* prices, int n_prices, double* out_returns) {
    for (int i = 1; i < n_prices; i++) {
        if (prices[i] > 1e-6 && prices[i-1] > 1e-6) {
            double ret = log(prices[i] / prices[i-1]);
            if (ret > 2.0) ret = 0.0;
            if (ret < -2.0) ret = 0.0;
            out_returns[i-1] = ret;
        } else {
            out_returns[i-1] = 0.0;
        }
    }
}

// --- Physics Constraints ---
int check_constraints(SVCJParams* p) {
    if (p->theta < 1e-7 || p->theta > 0.01) return 0; // Max 10% daily vol base
    if (p->v0 < 1e-7 || p->v0 > 0.02) return 0;
    if (p->kappa < 0.01 || p->kappa > 2.0) return 0;
    if (p->sigma_v < 1e-6 || p->sigma_v > 0.5) return 0;
    if (p->rho < -0.99 || p->rho > 0.99) return 0;
    return 1;
}

// --- Likelihood ---
double calculate_nll(double* returns, int n_steps, SVCJParams* p) {
    if (!check_constraints(p)) return INF_LOSS;
    
    double nll = 0.0;
    double vt = p->v0;
    // Assume a certain jump frequency for NLL cost
    double jump_var_contrib = p->lambda_j * (p->mu_j*p->mu_j + p->sigma_j*p->sigma_j);

    for (int i = 0; i < n_steps; i++) {
        if (vt < 1e-8) vt = 1e-8;
        if (vt > 0.05) vt = 0.05; // Cap at 22% daily vol for stability

        double drift = p->kappa * (p->theta - vt);
        double vt_pred = vt + drift;
        if (vt_pred < 1e-8) vt_pred = 1e-8;

        double total_var = vt_pred + jump_var_contrib;
        
        // Likelihood
        double prob = density_norm(returns[i], 0.0, total_var);
        if (prob < 1e-15) prob = 1e-15;
        nll -= log(prob);

        // Update
        double shock = (returns[i]*returns[i]) - total_var;
        vt = vt_pred + (p->sigma_v * 1.0) * shock; 
    }
    return nll;
}

// --- 1. Robust Optimizer (Median Initialization) ---
void optimize_params_history(double* returns, int n_steps, SVCJParams* out) {
    // A. Robust Estimator for Diffusive Vol (Median)
    // Allocating on stack for small history, or heap for safety
    double* sq_rets = (double*)malloc(n_steps * sizeof(double));
    int valid_count = 0;
    
    for(int i=0; i<n_steps; i++) {
        if(returns[i] != 0.0) {
            sq_rets[valid_count++] = returns[i] * returns[i];
        }
    }
    
    double median_var = 1e-4;
    if (valid_count > 0) {
        qsort(sq_rets, valid_count, sizeof(double), compare_doubles);
        median_var = sq_rets[valid_count / 2];
    }
    free(sq_rets);

    if (median_var < 1e-7) median_var = 1e-5;

    // B. Initialize (Force Theta to Median, not Mean)
    out->theta = median_var; 
    out->v0 = median_var;
    out->lambda_j = 0.05; 
    out->mu_j = -0.02;
    out->sigma_j = sqrt(median_var) * 5.0; // Jumps are 5x normal

    // C. Grid Search
    double best_nll = INF_LOSS;
    SVCJParams best_p = *out;
    SVCJParams candidate = *out;

    // We scan for Kappa (speed) and Sigma_V (volatility of volatility)
    // Low Sigma_V prevents the model from explaining jumps as "just high vol"
    double kappas[] = {0.05, 0.2, 0.5};
    double rhos[] = {-0.6, -0.2};
    double sigmas[] = {median_var * 0.1, median_var * 1.0, median_var * 3.0}; 

    for(int k=0; k<3; k++) {
        for(int r=0; r<2; r++) {
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

// --- 2. Robust UKF Filter (Jump Separation) ---
void run_ukf_filter(double* returns, int n_steps, SVCJParams* p, FilterState* out_states) {
    double vt = p->v0;
    
    for (int i = 0; i < n_steps; i++) {
        if (vt < 1e-8) vt = 1e-8;

        // A. Predict
        double drift = p->kappa * (p->theta - vt);
        double vt_pred = vt + drift;
        if (vt_pred < 1e-8) vt_pred = 1e-8;

        // B. Detect Jump (BEFORE Update)
        double innovation = returns[i];
        double diffusive_std = sqrt(vt_pred);
        double z_score = innovation / diffusive_std;
        double jump_prob = 0.0;

        // Lower threshold for sensitivity (2.5 sigma)
        if (fabs(z_score) > 2.5) {
            double power = fabs(z_score) - 2.5;
            if(power > 10) power = 10;
            jump_prob = 1.0 - exp(-power * 1.5); // Sigmoid
        }

        // C. Robust Update
        // If it's a jump, we reduce the weight of this observation on the continuous vol update.
        // weight = 1.0 (Pure Diffusion) -> 0.0 (Pure Jump)
        double diff_weight = 1.0 - jump_prob;
        if (diff_weight < 0.05) diff_weight = 0.05; // Keep a little adaptation

        // Kalman Gain
        double total_var = vt_pred + (p->lambda_j * p->sigma_j * p->sigma_j);
        double gain = (p->sigma_v * p->rho) / (total_var + 1e-9);

        // Clamp gain
        if (gain > 100.0) gain = 100.0;
        if (gain < -100.0) gain = -100.0;

        double innov_sq = innovation * innovation;
        
        // **KEY FIX**: Scale the update by diff_weight.
        // Don't let jumps spike the continuous vol.
        vt = vt_pred + (gain * (innov_sq - total_var) * diff_weight);
        
        // Bounds
        if (vt > p->theta * 10.0) vt = p->theta * 10.0;
        if (vt < 1e-8) vt = 1e-8;

        // Output
        out_states[i].spot_vol = sqrt(vt);
        out_states[i].jump_prob = jump_prob;
        out_states[i].drift_residue = innovation;
    }
}

// --- 3. Option Calibration ---
void calibrate_to_options(OptionContract* options, int n_opts, double spot, SVCJParams* out) {
    double sum_iv_sq = 0.0;
    int count = 0;
    double skew = 0.0;

    for(int i=0; i<n_opts; i++) {
        double m = options[i].strike / spot;
        if (m < 0.85 || m > 1.15) continue; // Strict Moneyness
        if (options[i].price < 0.01) continue;
        if (options[i].T_years < 0.04) continue; // > 2 weeks

        if (fabs(m - 1.0) < 0.05) {
            double iv = (options[i].price / spot * 2.5) / sqrt(options[i].T_years);
            sum_iv_sq += (iv * iv) / 252.0; // Convert to Daily
            count++;
        }
        if (m < 0.95 && !options[i].is_call) skew -= 1.0;
    }

    if (count > 0) {
        double imp_theta = sum_iv_sq / count;
        // Conservative blend: 50/50 to keep robust stats relevant
        out->theta = (out->theta * 0.5) + (imp_theta * 0.5);
        out->v0 = imp_theta; 
    }
    
    if (skew < -3.0) out->rho = -0.6;
}