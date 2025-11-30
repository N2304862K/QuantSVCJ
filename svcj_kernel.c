#include "svcj_kernel.h"
#include <stdio.h>
#include <float.h>

#define PI 3.14159265358979323846
#define MAX_ITER 15 // Optimization iterations

// --- Helper: Constraints ---
void enforce_feller(SVCJParams* p) {
    // 1. Positivity
    if (p->theta < 0.001) p->theta = 0.001;     // Min 3% vol
    if (p->v0 < 0.001) p->v0 = 0.001;
    if (p->sigma_v < 0.01) p->sigma_v = 0.01;
    
    // 2. Correlation Bound
    if (p->rho > 0.95) p->rho = 0.95;
    if (p->rho < -0.95) p->rho = -0.95;

    // 3. Feller Condition Soft Check (Stability > Exactness for trading)
    // 2*kappa*theta > sigma_v^2. If violated, variance can hit zero.
    // We bump kappa if needed.
    double rhs = p->sigma_v * p->sigma_v;
    double lhs = 2.0 * p->kappa * p->theta;
    if (lhs < rhs) {
        p->kappa = (rhs / (2.0 * p->theta)) + 0.5;
    }
    if (p->kappa > 20.0) p->kappa = 20.0; // Cap mean reversion
}

// --- Core 1: Likelihood Evaluator (UKF-based) ---
double calculate_neg_log_likelihood(double* returns, int n_steps, SVCJParams* p) {
    double nll = 0.0;
    double dt = 1.0 / 252.0;
    double vt = p->v0;
    
    // Pre-calc jump variance part
    double jump_var_contrib = p->lambda_j * (p->mu_j*p->mu_j + p->sigma_j*p->sigma_j) * dt;

    for (int i = 0; i < n_steps; i++) {
        // Prevent explosion
        if (vt > 5.0) vt = 5.0; 
        if (vt < 1e-4) vt = 1e-4;

        // 1. Predict
        double drift = p->kappa * (p->theta - vt);
        double vt_pred = vt + drift * dt;
        if (vt_pred < 1e-4) vt_pred = 1e-4;

        // 2. Measure (Total Variance = Continuous + Jumps)
        double tot_var = vt_pred * dt + jump_var_contrib;
        double std_dev = sqrt(tot_var);
        
        double innovation = returns[i]; // Assuming zero mean drift for volatility fitting simplicity
        double lik = (1.0 / (sqrt(2.0 * PI) * std_dev)) * exp(-0.5 * (innovation * innovation) / tot_var);
        
        if (lik < 1e-10) lik = 1e-10;
        nll -= log(lik);

        // 3. Update State (Simple innovation update for the filter loop)
        // Note: Full UKF update is in run_ukf_filter. Here we just need NLL estimate.
        // We use a simplified GARCH-like update for NLL speed.
        double shock = (innovation*innovation) - tot_var;
        vt = vt_pred + (p->kappa * dt * 0.1) * shock; // Heuristic adaptation
    }
    return nll;
}

// --- Core 2: Coordinate Descent Optimizer ---
void optimize_params_history(double* returns, int n_steps, SVCJParams* out) {
    // 1. Initialize with sane defaults
    // Estimate Variance
    double sum_sq = 0.0;
    for(int i=0; i<n_steps; i++) sum_sq += returns[i]*returns[i];
    double realized_var_ann = (sum_sq / n_steps) * 252.0;
    
    out->theta = realized_var_ann;
    out->v0 = realized_var_ann;
    out->kappa = 4.0;
    out->sigma_v = 0.4;
    out->rho = -0.7; // Equity usually negative skew
    out->lambda_j = 0.5; // Avg 1 jump every 2 years
    out->mu_j = -0.02;
    out->sigma_j = 0.05;

    // 2. Optimization Loop
    double best_nll = calculate_neg_log_likelihood(returns, n_steps, out);
    double learning_rate = 0.1;

    // We cycle through parameters and perturb them
    for (int iter = 0; iter < MAX_ITER; iter++) {
        SVCJParams current = *out;
        
        // Optimize Kappa
        double candidates_k[] = {current.kappa * 1.2, current.kappa * 0.8};
        for(int j=0; j<2; j++) {
            current.kappa = candidates_k[j];
            enforce_feller(&current);
            double nll = calculate_neg_log_likelihood(returns, n_steps, &current);
            if(nll < best_nll) { best_nll = nll; *out = current; }
            else current = *out; // Revert
        }

        // Optimize Sigma_v
        double candidates_s[] = {current.sigma_v + 0.05, current.sigma_v - 0.05};
        for(int j=0; j<2; j++) {
            current.sigma_v = candidates_s[j];
            enforce_feller(&current);
            double nll = calculate_neg_log_likelihood(returns, n_steps, &current);
            if(nll < best_nll) { best_nll = nll; *out = current; }
            else current = *out;
        }
        
        // Optimize Rho
        double candidates_r[] = {current.rho + 0.1, current.rho - 0.1};
        for(int j=0; j<2; j++) {
            current.rho = candidates_r[j];
            enforce_feller(&current);
            double nll = calculate_neg_log_likelihood(returns, n_steps, &current);
            if(nll < best_nll) { best_nll = nll; *out = current; }
            else current = *out;
        }

        // Optimize V0 (Fast convergence)
        current.v0 = current.v0 * 0.9 + (returns[n_steps-1]*returns[n_steps-1]*252.0) * 0.1;
        enforce_feller(&current);
        *out = current;
    }
}

// --- Core 3: UKF Run (Final Pass) ---
void run_ukf_filter(double* returns, int n_steps, SVCJParams* p, FilterState* out_states) {
    double vt = p->v0;
    double dt = 1.0 / 252.0;

    for (int i = 0; i < n_steps; i++) {
        double ret = returns[i];
        
        // Prediction
        double drift = p->kappa * (p->theta - vt);
        double vt_pred = vt + drift * dt;
        if (vt_pred < 1e-4) vt_pred = 1e-4;

        // Jump Diffusion Expectation
        double total_var = vt_pred * dt + p->lambda_j * (p->mu_j*p->mu_j + p->sigma_j*p->sigma_j) * dt;
        
        // Update (Kalman Gain Approximation)
        double innovation = ret * ret - total_var; // Variance innovation
        // Gain driven by Vol-of-Vol and Correlation
        double gain = (p->sigma_v * fabs(p->rho)) / (sqrt(vt_pred)*2.0); 
        
        vt = vt_pred + gain * innovation;
        
        // Clamp
        if (vt < 1e-4) vt = 1e-4;
        if (vt > 10.0) vt = 10.0;

        // Jump Detection (Z-Score)
        double z_score = ret / sqrt(total_var);
        double jump_prob = 0.0;
        
        // Sigmoid probability activation for jump
        if (fabs(z_score) > 2.5) {
            double exponent = fabs(z_score) - 2.5;
            jump_prob = 1.0 - exp(-exponent);
        }

        // Output
        out_states[i].vt = vt;
        out_states[i].spot_vol = sqrt(vt); // Annualized Vol
        out_states[i].jump_prob = jump_prob;
        out_states[i].drift_residue = ret - (vt * -0.5 * dt); // Jensen's inequality drift
    }
}

// --- Core 4: Option Calibration ---
void calibrate_to_options(OptionContract* options, int n_opts, double spot, SVCJParams* out) {
    // 1. Calculate Implied Vols from Market
    // Heuristic: If OTM Puts are expensive, Rho is more negative.
    // If ATM Vol is high, Theta is high.
    
    double atm_vol_sum = 0.0;
    int atm_count = 0;
    
    for(int i=0; i<n_opts; i++) {
        // ATM Check
        if(fabs(options[i].strike - spot)/spot < 0.02) {
            // Rough BS Inversion approx for speed (Price / (0.4 * S * sqrt(T)))
            double est_vol = options[i].price / (0.4 * spot * sqrt(options[i].T));
            atm_vol_sum += est_vol;
            atm_count++;
        }
    }
    
    if (atm_count > 0) {
        double mkt_vol = atm_vol_sum / atm_count;
        out->theta = mkt_vol * mkt_vol;
        out->v0 = out->theta;
        out->sigma_v = mkt_vol * 1.5; // Vol of vol typically scales with vol
    }
    
    // Adjust Rho based on skew (Mock logic: usually requires full fit)
    out->rho = -0.6; // Default to sticky strike
    enforce_feller(out);
}