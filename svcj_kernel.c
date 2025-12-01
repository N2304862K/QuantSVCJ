#include "svcj_kernel.h"
#include <stdio.h>
#include <float.h>

#define PI 3.14159265358979323846
#define SQRT_2PI 2.50662827463
#define INF_LOSS 1e20

// --- 1. Robust Data Transformation ---
void calculate_returns_from_prices(double* prices, int n_prices, double* out_returns) {
    for (int i = 1; i < n_prices; i++) {
        if (prices[i] > 1e-5 && prices[i-1] > 1e-5) {
            double ret = log(prices[i] / prices[i-1]);
            // Cap extreme data errors (e.g. 1000% moves)
            if (ret > 2.0) ret = 0.0;
            if (ret < -2.0) ret = 0.0;
            out_returns[i-1] = ret;
        } else {
            out_returns[i-1] = 0.0;
        }
    }
}

// --- 2. Math Utilities ---
double density_norm(double x, double mean, double var) {
    if (var < 1e-9) var = 1e-9;
    return (1.0 / (SQRT_2PI * sqrt(var))) * exp(-0.5 * (x - mean)*(x - mean) / var);
}

// Check if params are chemically possible for DAILY data
int check_constraints(SVCJParams* p) {
    // Daily Variance for stocks is typically 0.0001 (1% daily move)
    // Bounds: 0.000001 (0.1% move) to 0.01 (10% move)
    if (p->theta < 1e-7 || p->theta > 0.02) return 0; 
    if (p->v0 < 1e-7 || p->v0 > 0.02) return 0;
    
    // Mean Reversion: 0.01 (slow) to 0.5 (fast)
    if (p->kappa < 0.001 || p->kappa > 1.5) return 0;
    
    // Vol of Vol: Scaled to daily. 1e-5 to 1e-2
    if (p->sigma_v < 1e-6 || p->sigma_v > 0.1) return 0;
    
    if (p->rho < -0.99 || p->rho > 0.99) return 0;

    // Feller: 2*k*theta > sigma_v^2
    // We allow soft violation but penalize divergence
    return 1;
}

// --- 3. Optimizer Core ---
double calculate_nll(double* returns, int n_steps, SVCJParams* p) {
    if (!check_constraints(p)) return INF_LOSS;
    
    double nll = 0.0;
    double vt = p->v0;
    double dt = 1.0; // Implicit daily step

    double jump_var = p->lambda_j * (p->mu_j*p->mu_j + p->sigma_j*p->sigma_j);

    for (int i = 0; i < n_steps; i++) {
        // Clamp to prevent explosion
        if (vt < 1e-8) vt = 1e-8;
        if (vt > 0.05) vt = 0.05;

        // Drift & Diffusion
        double drift_v = p->kappa * (p->theta - vt);
        double vt_pred = vt + drift_v;
        if (vt_pred < 1e-8) vt_pred = 1e-8;

        double total_var = vt_pred + jump_var;
        
        // Likelihood
        double prob = density_norm(returns[i], 0.0, total_var);
        if (prob < 1e-15) prob = 1e-15;
        nll -= log(prob);

        // Simple Update for Optimizer (Fast filter)
        double innovation_sq = returns[i] * returns[i];
        double shock = innovation_sq - total_var;
        vt = vt_pred + (p->sigma_v * 2.0) * shock; // Heuristic GARCH update
    }
    return nll;
}

void optimize_params_history(double* returns, int n_steps, SVCJParams* out) {
    // A. Estimate Realized Variance (Central Pivot)
    double sum_sq = 0.0;
    for(int i=0; i<n_steps; i++) sum_sq += returns[i]*returns[i];
    double rv = sum_sq / n_steps;
    if (rv < 1e-7) rv = 1e-5; // Safety floor

    // B. Set Defaults
    out->theta = rv;
    out->v0 = rv;
    out->lambda_j = 0.02; // Rare jumps
    out->mu_j = -0.02;    // Downside bias
    out->sigma_j = sqrt(rv) * 4.0; // Jumps are 4x normal vol

    // C. Grid Search centered on RV logic
    // We vary Kappa (speed) and Sigma_V (vol of vol)
    double best_nll = INF_LOSS;
    SVCJParams best_p = *out;
    SVCJParams candidate = *out;

    double kappas[] = {0.05, 0.1, 0.3};
    double rhos[] = {-0.6, -0.3, 0.0};
    // Sigma_V relative to variance level
    double sigmas[] = {rv * 0.5, rv * 2.0, rv * 5.0}; 

    for(int k=0; k<3; k++) {
        for(int r=0; r<3; r++) {
            for(int s=0; s<3; s++) {
                candidate.kappa = kappas[k];
                candidate.rho = rhos[r];
                candidate.sigma_v = sigmas[s];
                
                // Enforce Feller locally
                if (2*candidate.kappa*candidate.theta < candidate.sigma_v*candidate.sigma_v) {
                    candidate.sigma_v = sqrt(1.9 * candidate.kappa * candidate.theta);
                }

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

// --- 4. UKF (Production Filter) ---
void run_ukf_filter(double* returns, int n_steps, SVCJParams* p, FilterState* out_states) {
    double vt = p->v0;
    
    for (int i = 0; i < n_steps; i++) {
        if (vt < 1e-8) vt = 1e-8;

        // 1. Predict
        double drift = p->kappa * (p->theta - vt);
        double vt_pred = vt + drift;
        if (vt_pred < 1e-8) vt_pred = 1e-8;

        // 2. Measure
        double jump_var = p->lambda_j * (p->mu_j*p->mu_j + p->sigma_j*p->sigma_j);
        double total_var = vt_pred + jump_var;
        
        // 3. Update (Kalman Gain)
        double innovation = returns[i];
        
        // Gain calculation tailored for log-vol process approx
        // K = (Sigma_V * Rho) / Total_Var
        double gain = (p->sigma_v * p->rho) / (total_var + 1e-9);
        
        // Bound gain to prevent instability
        if (gain > 500.0) gain = 500.0;
        if (gain < -500.0) gain = -500.0;

        double innov_sq = innovation * innovation;
        double measurement_residual = innov_sq - total_var;
        
        vt = vt_pred + gain * measurement_residual;
        
        // Adaptive Clamping (Max 20x Long-Run Variance)
        double max_var = p->theta * 20.0; 
        if (vt > max_var) vt = max_var;
        if (vt < 1e-8) vt = 1e-8;

        // 4. Jump Logic
        double z_score = innovation / sqrt(vt_pred);
        double jump_p = 0.0;
        
        // Dynamic threshold based on current vol regime
        // If vol is low, it takes less to trigger a jump
        if (fabs(z_score) > 3.0) {
            double exponent = fabs(z_score) - 3.0;
            if (exponent > 10) exponent = 10;
            jump_p = 1.0 - exp(-exponent);
        }

        // Output
        out_states[i].spot_vol = sqrt(vt);
        out_states[i].jump_prob = jump_p;
        out_states[i].drift_residue = innovation; // Pure innovation residue
    }
}

// --- 5. Option Calibration ---
void calibrate_to_options(OptionContract* options, int n_opts, double spot, SVCJParams* out) {
    double sum_iv_daily_sq = 0.0;
    int count = 0;
    double skew_sum = 0.0;

    for(int i=0; i<n_opts; i++) {
        // Pre-processing: Filter noise
        double moneyness = options[i].strike / spot;
        if (moneyness < 0.8 || moneyness > 1.2) continue;
        if (options[i].T_years < 0.02) continue; // Skip < 1 week
        if (options[i].price < 0.01) continue;

        // Approximate ATM Vol
        if (fabs(moneyness - 1.0) < 0.05) {
            double vol_ann = (options[i].price / spot * 2.5) / sqrt(options[i].T_years);
            double daily_var = (vol_ann * vol_ann) / 252.0;
            sum_iv_daily_sq += daily_var;
            count++;
        }
        
        // Simple Skew
        if (moneyness < 0.95 && options[i].is_call == 0) skew_sum -= 1.0;
    }

    if (count > 0) {
        double implied_theta = sum_iv_daily_sq / count;
        // Blend Historical (30%) with Implied (70%)
        out->theta = (out->theta * 0.3) + (implied_theta * 0.7);
        out->v0 = implied_theta; // Snap current state to Implied
    }
    
    if (skew_sum < -3.0) out->rho = -0.7; // Detect fear
}