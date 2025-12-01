#include "svcj_kernel.h"
#include <stdio.h>
#include <float.h>

#define PI 3.14159265358979323846
#define SQRT_2PI 2.50662827463
#define INF_LOSS 1e15

double gaussian_pdf(double x, double mean, double var) {
    if (var < 1e-9) var = 1e-9;
    return (1.0 / (SQRT_2PI * sqrt(var))) * exp(-0.5 * (x - mean)*(x - mean) / var);
}

// --- Likelihood (Including Drift) ---
double calculate_nll(double* returns, int n_steps, SVCJParams* p) {
    double nll = 0.0;
    double vt = p->v0;

    for (int i = 0; i < n_steps; i++) {
        if (vt < 1e-8) vt = 1e-8;
        if (vt > 0.01) vt = 0.01; // Clamp max daily var (~158% ann vol)

        // 1. Expected Drift (Convexity Adjusted)
        // E[r] = alpha - 0.5*vt (Geometric Brownian Motion correction)
        double mu_diff = p->alpha - 0.5 * vt;
        
        // 2. Diffusive Density
        double pdf_diff = gaussian_pdf(returns[i], mu_diff, vt);
        
        // 3. Jump Density
        // Jump Mean = Drift + Jump Mean Size
        // Jump Var = Diffusive Var + Jump Var
        double mu_jump = mu_diff + p->mu_j;
        double var_jump = vt + (p->sigma_j * p->sigma_j);
        double pdf_jump = gaussian_pdf(returns[i], mu_jump, var_jump);

        // 4. Mixture
        double lik = (1.0 - p->lambda_j) * pdf_diff + p->lambda_j * pdf_jump;
        
        if (lik < 1e-15) lik = 1e-15;
        nll -= log(lik);

        // 5. Update State (Fast GARCH for Optimization Loop)
        double resid = returns[i] - mu_diff;
        vt = p->theta * 0.05 + 0.90 * vt + 0.05 * (resid * resid); 
    }
    return nll;
}

// --- Optimizer ---
void optimize_params_history(double* returns, int n_steps, SVCJParams* out) {
    // 1. Basic Stats (Daily)
    double sum = 0.0;
    double sum_sq = 0.0;
    for(int i=0; i<n_steps; i++) {
        sum += returns[i];
        sum_sq += returns[i]*returns[i];
    }
    double mean = sum / n_steps;
    double var = (sum_sq / n_steps) - (mean * mean);
    
    // Defaults (Daily Units)
    out->alpha = mean;
    out->theta = var;
    out->v0 = var;
    out->mu_j = -0.01;      // -1% average jump
    out->sigma_j = sqrt(var) * 3.0; 
    out->kappa = 0.1;
    out->rho = -0.5;
    
    // 2. Grid Search (Focus on Jump Parameters)
    double lambdas[] = {0.002, 0.02, 0.10}; // 0.2%, 2%, 10% daily jump prob
    double sigma_vs[] = {0.05, 0.15};
    
    double best_nll = INF_LOSS;
    SVCJParams best_p = *out;
    SVCJParams candidate = *out;

    for(int l=0; l<3; l++) {
        for(int s=0; s<2; s++) {
            candidate.lambda_j = lambdas[l];
            candidate.sigma_v = sigma_vs[s];
            
            double nll = calculate_nll(returns, n_steps, &candidate);
            if (nll < best_nll) {
                best_nll = nll;
                best_p = candidate;
            }
        }
    }
    
    // 3. Quick Polish on Alpha (Drift)
    // Simple gradient step
    double d_alpha = 0.0001;
    SVCJParams up = best_p; up.alpha += d_alpha;
    SVCJParams dn = best_p; dn.alpha -= d_alpha;
    if (calculate_nll(returns, n_steps, &up) < best_nll) best_p = up;
    else if (calculate_nll(returns, n_steps, &dn) < best_nll) best_p = dn;

    *out = best_p;
}

// --- UKF Filter (State Extraction) ---
void run_ukf_filter(double* returns, int n_steps, SVCJParams* p, FilterState* out_states) {
    double vt = p->v0;
    
    for (int i = 0; i < n_steps; i++) {
        if (vt < 1e-8) vt = 1e-8;
        if (vt > 0.02) vt = 0.02;

        // 1. Calculate Forecast for CURRENT step i (based on vt from i-1)
        double drift_geo = p->alpha - 0.5 * vt; // Geometric drift correction
        double expected_jump = p->lambda_j * p->mu_j;
        double forecast = drift_geo + expected_jump;

        // 2. Densities for Probability
        double pdf_diff = gaussian_pdf(returns[i], drift_geo, vt);
        double pdf_jump = gaussian_pdf(returns[i], drift_geo + p->mu_j, vt + p->sigma_j*p->sigma_j);

        // 3. Jump Probability
        double likelihood_ratio = (p->lambda_j * pdf_jump) / 
                                  ((1.0 - p->lambda_j) * pdf_diff + p->lambda_j * pdf_jump + 1e-20);
        
        if (likelihood_ratio > 0.999) likelihood_ratio = 0.999;
        
        // 4. State Update (Bayesian weighting)
        double innovation = returns[i] - drift_geo;
        double innovation_sq = innovation * innovation;
        
        // If jump, we discount the innovation's impact on continuous variance
        double update_weight = (1.0 - likelihood_ratio); 
        
        // Forecast for NEXT step
        double vt_next = 0.94 * vt + 0.04 * p->theta + 0.02 * (innovation_sq * update_weight);

        // Store
        out_states[i].vt = vt;
        out_states[i].spot_vol = sqrt(vt); // Daily Vol
        out_states[i].jump_prob = likelihood_ratio;
        out_states[i].drift_residue = returns[i] - forecast;
        
        // Horizon Forecast: Forecast for i+1 using updated vt_next
        double next_drift_geo = p->alpha - 0.5 * vt_next;
        out_states[i].drift_forecast = next_drift_geo + expected_jump;
        
        vt = vt_next;
    }
}

// --- Option Calibration ---
void calibrate_to_options(OptionContract* options, int n_opts, double spot, SVCJParams* out) {
    if (n_opts == 0) return;
    double sum_imp_var = 0.0;
    int count = 0;

    for(int i=0; i<n_opts; i++) {
        if(fabs(options[i].strike - spot)/spot < 0.05) {
            // BS Inversion approx
            double vol_ann = (options[i].price / spot) * (1.0 / (0.4 * sqrt(options[i].T)));
            sum_imp_var += vol_ann * vol_ann;
            count++;
        }
    }

    if (count > 0) {
        double ann_var = sum_imp_var / count;
        // CONVERT ANNUALIZED TO DAILY
        double daily_var = ann_var / 252.0;
        
        out->theta = daily_var;
        out->v0 = daily_var; // Set current state to implied level
        
        if (daily_var > 0.0004) { // >30% Ann Vol
            out->lambda_j = 0.1; 
        }
    }
}