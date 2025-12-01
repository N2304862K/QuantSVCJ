#include "svcj.h"
#include <stdio.h>

// --- Helper: Zero-Return Jitter/Denoising ---
void clean_returns(double* returns, int n, int stride) {
    for(int i=0; i<n; i++) {
        // Access via stride to jump rows if needed
        double val = returns[i * stride];
        if(fabs(val) < JITTER_THRESHOLD) {
            double noise = (i % 2 == 0) ? JITTER_THRESHOLD : -JITTER_THRESHOLD;
            returns[i * stride] = noise;
        }
    }
}

// --- Helper: Stability Constraints (Daily Scale Adjusted) ---
void check_feller_and_fix(SVCJParams* p) {
    // 1. Hard Positivity Boundaries (Scaled down for Daily values)
    if(p->kappa < 1e-5) p->kappa = 1e-5;
    if(p->theta < 1e-8) p->theta = 1e-8;  // Daily var can be very small (e.g. 1e-5)
    if(p->sigma_v < 1e-5) p->sigma_v = 1e-5;
    if(p->lambda_j < 0) p->lambda_j = 1e-6;
    if(p->sigma_j < 1e-6) p->sigma_j = 1e-6;

    // 2. Correlation Bound
    if(p->rho > 0.99) p->rho = 0.99;
    if(p->rho < -0.99) p->rho = -0.99;

    // 3. Feller Condition: 2 * kappa * theta > sigma_v^2
    double feller_boundary = 2.0 * p->kappa * p->theta;
    if (p->sigma_v * p->sigma_v >= feller_boundary) {
        p->sigma_v = sqrt(feller_boundary * 0.95);
    }
}

// --- Core: Unscented Kalman Filter Step (Simplified for SVCJ) ---
double run_ukf_qmle(double* returns, int n, int stride, SVCJParams* p, double* out_spot_vol, double* out_jump_prob) {
    double log_likelihood = 0.0;
    
    // Initialize State (Variance)
    double v_curr = p->theta; 
    
    for(int t=0; t<n; t++) {
        // Access data with stride
        double ret_t = returns[t * stride];

        // 1. Prediction Step
        double v_pred = v_curr + p->kappa * (p->theta - v_curr) * DT;
        if(v_pred < 1e-8) v_pred = 1e-8; // Safety floor

        // 2. Innovation / Observation
        double expected_ret = (p->mu - 0.5 * v_pred) * DT; 
        double y_tilde = ret_t - expected_ret;
        
        // 3. Update Step 
        double h_variance = v_pred * DT + (p->lambda_j * (p->mu_j*p->mu_j + p->sigma_j*p->sigma_j)) * DT;
        
        // Avoid division by zero
        if(h_variance < 1e-9) h_variance = 1e-9;

        double kalman_gain = (p->rho * p->sigma_v * DT) / h_variance;
        
        v_curr = v_pred + kalman_gain * y_tilde;
        
        // --- CRITICAL STABILITY FIX: Prevent Negative Variance ---
        if(v_curr < 1e-8) v_curr = 1e-8;
        
        // 4. Calculate Jump Prob 
        double jump_metric = (y_tilde * y_tilde) / h_variance;
        double j_prob = (jump_metric > 4.0) ? 1.0 : (p->lambda_j * DT); 
        
        // Store outputs
        if(out_spot_vol) out_spot_vol[t] = sqrt(v_curr);
        if(out_jump_prob) out_jump_prob[t] = j_prob;

        // QMLE Accumulation
        log_likelihood += -0.5 * log(2 * M_PI * h_variance) - 0.5 * (y_tilde * y_tilde) / h_variance;
    }

    return -log_likelihood; 
}

// --- Option Pricing (BSM Proxy for Demo) ---
void price_option_chain(double s0, double* strikes, double* expiries, int* types, int n_opts, SVCJParams* p, double* out_prices) {
    for(int i=0; i<n_opts; i++) {
        double K = strikes[i];
        double T = expiries[i]; // T is in Years
        
        // Scale Daily Params to Annualized for Pricing
        double theta_ann = p->theta * 252.0;
        double kappa_ann = p->kappa * 252.0;
        double mu_ann = p->mu * 252.0;
        double lambda_ann = p->lambda_j * 252.0;

        // Expected integrated variance
        double expected_var = theta_ann + (theta_ann - theta_ann) * ((1 - exp(-kappa_ann*T))/(kappa_ann*T)); 
        double total_vol = sqrt(expected_var + lambda_ann * (p->mu_j*p->mu_j + p->sigma_j*p->sigma_j));
        
        if (total_vol < 1e-4) total_vol = 1e-4;

        double d1 = (log(s0/K) + (mu_ann + 0.5*total_vol*total_vol)*T) / (total_vol*sqrt(T));
        // double d2 = d1 - total_vol*sqrt(T);
        
        // Intrinsic + Time Value Proxy
        double intrinsic = (types[i] == 1) ? (s0 - K) : (K - s0); 
        if(intrinsic < 0) intrinsic = 0;
        
        out_prices[i] = intrinsic + s0 * 0.1 * total_vol * sqrt(T); 
    }
}