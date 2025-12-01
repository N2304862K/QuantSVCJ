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
            // Hard Cap: Remove data errors (e.g. split artifacts > 100%)
            if (ret > 1.0) ret = 0.0;
            if (ret < -1.0) ret = 0.0;
            out_returns[i-1] = ret;
        } else {
            out_returns[i-1] = 0.0;
        }
    }
}

// --- STRICT Constraints ---
int check_constraints(SVCJParams* p) {
    if (p->theta < 1e-7 || p->theta > 0.01) return 0; // Max 10% daily base vol
    if (p->v0 < 1e-7 || p->v0 > 0.02) return 0;
    
    // STRICT KAPPA: Fast mean reversion helps pull vol back after a jump
    if (p->kappa < 0.1 || p->kappa > 5.0) return 0; 
    
    // STRICT SIGMA_V: Cap vol-of-vol to prevent "eating" jumps
    // Max 0.1 means vol cannot change drastically in one day
    if (p->sigma_v < 1e-6 || p->sigma_v > 0.1) return 0; 

    if (p->rho < -0.99 || p->rho > 0.99) return 0;
    return 1;
}

// --- Penalized Likelihood ---
double calculate_nll(double* returns, int n_steps, SVCJParams* p) {
    if (!check_constraints(p)) return INF_LOSS;
    
    double nll = 0.0;
    double vt = p->v0;
    
    // Explicit Jump Variance component
    double jump_var_contrib = p->lambda_j * (p->mu_j*p->mu_j + p->sigma_j*p->sigma_j);

    for (int i = 0; i < n_steps; i++) {
        if (vt < 1e-8) vt = 1e-8;
        if (vt > 0.02) vt = 0.02; // Cap continuous variance at ~14% daily

        double drift = p->kappa * (p->theta - vt);
        double vt_pred = vt + drift;
        if (vt_pred < 1e-8) vt_pred = 1e-8;

        double total_var = vt_pred + jump_var_contrib;
        double lik = density_norm(returns[i], 0.0, total_var);
        
        if (lik < 1e-10) lik = 1e-10;
        nll -= log(lik);

        // Simple GARCH update for likelihood estimation
        double shock = (returns[i]*returns[i]) - total_var;
        vt = vt_pred + (p->sigma_v * 2.0) * shock; 
    }

    // *** PENALTY TERM ***
    // We penalize high Sigma_V. This forces the optimizer to choose Jumps
    // to explain outliers, rather than cranking up volatility flexibility.
    nll += (p->sigma_v * 5000.0); 

    return nll;
}

void optimize_params_history(double* returns, int n_steps, SVCJParams* out) {
    // 1. Robust Median Variance (The "Diffusive" Baseline)
    double* sq_rets = (double*)malloc(n_steps * sizeof(double));
    int valid_count = 0;
    for(int i=0; i<n_steps; i++) {
        if(returns[i] != 0.0) sq_rets[valid_count++] = returns[i] * returns[i];
    }
    double median_var = 1e-4;
    if (valid_count > 0) {
        qsort(sq_rets, valid_count, sizeof(double), compare_doubles);
        median_var = sq_rets[valid_count / 2];
    }
    free(sq_rets);

    // 2. Set strict priors favoring jumps
    out->theta = median_var;
    out->v0 = median_var;
    out->lambda_j = 0.1; // Expect jumps
    out->mu_j = -0.02;
    out->sigma_j = sqrt(median_var) * 4.0; 

    // 3. Grid Search (Focus on low Sigma_V)
    double best_nll = INF_LOSS;
    SVCJParams best_p = *out;
    SVCJParams candidate = *out;

    double kappas[] = {0.5, 1.5, 3.0};
    double rhos[] = {-0.7, -0.4};
    // Very constrained Sigma_V (Low vol-of-vol)
    double sigmas[] = {1e-4, 1e-3, 0.01}; 

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

// --- The "Judge" Filter ---
void run_ukf_filter(double* returns, int n_steps, SVCJParams* p, FilterState* out_states) {
    double vt = p->v0;
    
    for (int i = 0; i < n_steps; i++) {
        if (vt < 1e-8) vt = 1e-8;

        // Predict
        double drift = p->kappa * (p->theta - vt);
        double vt_pred = vt + drift;
        if (vt_pred < 1e-8) vt_pred = 1e-8;

        // JUMP DETECTION (The Judge)
        // We use a "Hybrid Volatility" for detection:
        // Geometric Mean of (Current State, Long Run Theta)
        // This anchors expectations. If vt exploded to 10%, but theta is 1%,
        // we judge jumps based on ~3% vol, not 10%.
        double judge_vol = sqrt(vt_pred) * 0.6 + sqrt(p->theta) * 0.4;
        double z_score = returns[i] / judge_vol;
        
        double jump_prob = 0.0;
        
        // Strict 3-sigma detection
        if (fabs(z_score) > 3.0) {
            double power = fabs(z_score) - 3.0;
            if(power > 10) power = 10;
            jump_prob = 1.0 - exp(-power);
        }

        // UPDATE
        double innovation = returns[i];
        double total_var = vt_pred + (p->lambda_j * p->sigma_j * p->sigma_j);
        double gain = (p->sigma_v * p->rho) / (total_var + 1e-9);

        // Filter weighting
        // If it's a jump, the continuous vol should NOT update much.
        // weight = 1.0 (Diffusion) -> 0.0 (Jump)
        double diffusion_weight = 1.0 - jump_prob;
        if (diffusion_weight < 0.0) diffusion_weight = 0.0;

        double innov_sq = innovation * innovation;
        
        // Only update 'vt' based on the 'diffusive' portion of the return
        vt = vt_pred + (gain * (innov_sq - total_var) * diffusion_weight);
        
        // Clamp
        if (vt > 0.02) vt = 0.02; // Max 14% Daily
        if (vt < 1e-8) vt = 1e-8;

        out_states[i].spot_vol = sqrt(vt);
        out_states[i].jump_prob = jump_prob;
        out_states[i].drift_residue = innovation;
    }
}

// --- Option Calib ---
void calibrate_to_options(OptionContract* options, int n_opts, double spot, SVCJParams* out) {
    // Standard calibration logic...
    double sum_iv = 0.0;
    int count = 0;
    
    for(int i=0; i<n_opts; i++) {
        double m = options[i].strike / spot;
        if (m < 0.85 || m > 1.15) continue;
        if (options[i].price < 0.01) continue;
        
        if (fabs(m - 1.0) < 0.05) {
            double iv = (options[i].price / spot * 2.5) / sqrt(options[i].T_years);
            double var = (iv*iv)/252.0;
            sum_iv += var;
            count++;
        }
    }
    
    // Only adjust Theta/V0 slightly to market levels
    // We trust our historical Median vol for the dynamics, 
    // we only trust options for the "Level".
    if (count > 0) {
        double implied_level = sum_iv / count;
        out->theta = implied_level; // Set long run anchor to implied
        // Don't overwrite V0 completely, blend it
        out->v0 = (out->v0 + implied_level) * 0.5;
    }
}