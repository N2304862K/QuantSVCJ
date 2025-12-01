#include "svcj_kernel.h"
#include <stdio.h>
#include <float.h>

#define PI 3.14159265358979323846
#define SQRT_2PI 2.50662827463
#define INF_LOSS 1e15

// --- Gaussian Density Helper ---
double gaussian_pdf(double x, double mean, double var) {
    if (var < 1e-9) var = 1e-9;
    return (1.0 / (SQRT_2PI * sqrt(var))) * exp(-0.5 * (x - mean)*(x - mean) / var);
}

// --- Likelihood Calculation (Bernoulli Mixture) ---
double calculate_nll(double* returns, int n_steps, SVCJParams* p) {
    double nll = 0.0;
    double vt = p->v0;
    double dt = 1.0; // Working in daily steps internally

    for (int i = 0; i < n_steps; i++) {
        // Prevent variance collapse or explosion
        if (vt < 1e-7) vt = 1e-7;
        if (vt > 0.005) vt = 0.005; // Cap at ~110% Annualized Vol to prevent explosion

        // 1. Diffusion Density (No Jump)
        double pdf_diff = gaussian_pdf(returns[i], 0.0, vt);
        
        // 2. Jump Density (Diffusion + Jump Variance)
        double jump_var = vt + (p->sigma_j * p->sigma_j);
        double pdf_jump = gaussian_pdf(returns[i], p->mu_j, jump_var);

        // 3. Total Probability (Mixture)
        // Prob = (1 - lambda) * Diff + lambda * Jump
        double lik = (1.0 - p->lambda_j) * pdf_diff + p->lambda_j * pdf_jump;
        
        if (lik < 1e-15) lik = 1e-15;
        nll -= log(lik);

        // 4. Variance Update (GARCH Approximation for fitting)
        // Simple GARCH(1,1) style update to propagate state
        double innovation2 = returns[i] * returns[i];
        vt = p->theta * 0.05 + 0.90 * vt + 0.05 * innovation2; 
    }
    return nll;
}

// --- Optimizer: Constrained Grid Search ---
void optimize_params_history(double* returns, int n_steps, SVCJParams* out) {
    // 1. Initial Variance Estimate (Daily)
    double sum_sq = 0.0;
    for(int i=0; i<n_steps; i++) sum_sq += returns[i]*returns[i];
    double daily_var = sum_sq / n_steps;
    
    // Defaults
    out->theta = daily_var;
    out->v0 = daily_var;
    out->mu_j = -0.02;     // Bias towards crashes
    out->sigma_j = sqrt(daily_var) * 4.0; // Jumps are large events
    out->kappa = 0.1;      // Slow mean reversion
    out->rho = -0.5;
    out->sigma_v = 0.1;

    // 2. Grid Search to force Jump Recognition
    // We scan Lambda (Jump Freq) and Sigma_J (Jump Size)
    double lambdas[] = {0.005, 0.05, 0.15}; // Rare vs Frequent
    double sigma_vs[] = {0.05, 0.2};        // Low vs High Vol-Vol
    
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
    
    *out = best_p;
}

// --- Core Filter: Bayesian Update ---
void run_ukf_filter(double* returns, int n_steps, SVCJParams* p, FilterState* out_states) {
    double vt = p->v0;
    
    for (int i = 0; i < n_steps; i++) {
        if (vt < 1e-7) vt = 1e-7;
        if (vt > 0.005) vt = 0.005;

        // 1. Densities
        double pdf_diff = gaussian_pdf(returns[i], 0.0, vt);
        double jump_var = vt + (p->sigma_j * p->sigma_j);
        double pdf_jump = gaussian_pdf(returns[i], p->mu_j, jump_var);

        // 2. Posterior Jump Probability (Bayes Rule)
        // P(Jump | Data) = (P(Data|Jump) * P(Jump)) / P(Data)
        double likelihood_ratio = (p->lambda_j * pdf_jump) / 
                                  ((1.0 - p->lambda_j) * pdf_diff + p->lambda_j * pdf_jump + 1e-20);
        
        if (likelihood_ratio > 1.0) likelihood_ratio = 1.0;
        
        // 3. Volatility Update
        // If it was a jump, variance shouldn't spike as much as diffusion
        // We weight the innovation by (1 - jump_prob)
        double innovation_sq = returns[i] * returns[i];
        double update_weight = (1.0 - likelihood_ratio); 
        
        // Adaptive update (GARCH-like with Jump filtering)
        vt = 0.94 * vt + 0.04 * p->theta + 0.02 * (innovation_sq * update_weight);

        // Output Conversion
        out_states[i].vt = vt;
        out_states[i].spot_vol = sqrt(vt * 252.0); // Annualize for display
        out_states[i].jump_prob = likelihood_ratio;
        out_states[i].drift_residue = returns[i]; 
    }
}

// --- Option Calibration ---
void calibrate_to_options(OptionContract* options, int n_opts, double spot, SVCJParams* out) {
    if (n_opts == 0) return;
    double sum_imp_var = 0.0;
    int count = 0;

    for(int i=0; i<n_opts; i++) {
        // ATM Approximation
        if(fabs(options[i].strike - spot)/spot < 0.05) {
            // Approx implied vol
            double vol = (options[i].price / spot) * (1.0 / (0.4 * sqrt(options[i].T)));
            sum_imp_var += vol * vol;
            count++;
        }
    }

    if (count > 0) {
        double ann_var = sum_imp_var / count;
        // Convert Annualized Variance to Daily Variance for the C-Kernel
        out->theta = ann_var / 252.0;
        out->v0 = out->theta;
        
        // If Options imply high variance, jumps are priced in
        if (ann_var > 0.04) { // > 20% Vol
            out->lambda_j = 0.1; // Increase jump fear
        }
    }
}