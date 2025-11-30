#include "svcj_kernel.h"
#include <stdio.h>
#include <float.h>

#define PI 3.14159265358979323846

// --- Helper: Denoising ---
void denoise_returns(double* returns, int n_steps, double jitter) {
    for (int i = 0; i < n_steps; i++) {
        if (fabs(returns[i]) < 1e-8) {
            returns[i] = (i % 2 == 0) ? jitter : -jitter;
        }
    }
}

// --- Helper: Feller Condition & Positivity ---
void enforce_feller(SVCJParams* p) {
    // Hard positivity
    if (p->theta < 1e-4) p->theta = 1e-4;
    if (p->sigma_v < 1e-4) p->sigma_v = 1e-4;
    if (p->v0 < 1e-4) p->v0 = 1e-4;
    
    // Feller: 2*kappa*theta > sigma_v^2
    double feller_boundary = (p->sigma_v * p->sigma_v) / (2.0 * p->theta);
    if (p->kappa < feller_boundary) {
        p->kappa = feller_boundary + 0.1; // Push into stable region
    }
    
    // Correlation bound
    if (p->rho > 0.99) p->rho = 0.99;
    if (p->rho < -0.99) p->rho = -0.99;
}

// --- Core 1: Unscented Kalman Filter (Simplified for SVCJ) ---
void run_ukf_filter(double* returns, int n_steps, SVCJParams* p, FilterState* out_states) {
    double vt = p->v0;
    double dt = 1.0 / 252.0; // Daily steps
    
    for (int i = 0; i < n_steps; i++) {
        double ret = returns[i];
        
        // 1. Prediction Step (Euler-Maruyama expectation)
        double drift = p->kappa * (p->theta - vt);
        double vt_pred = vt + drift * dt;
        if (vt_pred < 1e-5) vt_pred = 1e-5;
        
        // 2. Innovation / Measurement
        // Expected return variance ~ vt + jumps
        double expected_var = vt_pred * dt + p->lambda_j * (p->mu_j * p->mu_j + p->sigma_j * p->sigma_j);
        double innovation = ret * ret - expected_var;
        
        // 3. Update Step (Simplified Kalman Gain for Vol)
        double gain = (p->sigma_v * p->rho) / (sqrt(expected_var) + 1e-6);
        vt = vt_pred + gain * innovation;
        if (vt < 1e-5) vt = 1e-5;

        // 4. Jump Probability (Bayesian update based on tail magnitude)
        double jump_likelihood = 0.0;
        double z_score = ret / sqrt(vt * dt);
        if (fabs(z_score) > 3.0) {
            jump_likelihood = 1.0 - exp(-0.5 * z_score * z_score);
        }

        // Store States
        out_states[i].spot_vol = sqrt(vt);
        out_states[i].jump_prob = jump_likelihood;
        out_states[i].drift_residue = ret - (vt * -0.5 * dt); // Simple drift correction
    }
}

// --- Core 2: Optimization (Placeholder for coordinate descent) ---
void optimize_params_history(double* returns, int n_steps, SVCJParams* out) {
    // Initial Guess
    out->kappa = 2.0; out->theta = 0.04; out->sigma_v = 0.3; 
    out->rho = -0.5; out->lambda_j = 0.1; out->mu_j = -0.05; 
    out->sigma_j = 0.1; out->v0 = 0.04;

    denoise_returns(returns, n_steps, 1e-5);
    
    // In a real scenario, this would loop calculate_neg_log_likelihood 
    // and adjust parameters via Gradient Descent or Nelder-Mead.
    // For this engine, we enforce constraints and perform a static fit approximation.
    enforce_feller(out);
}

// --- Core 3: Option Calibration ---
void calibrate_to_options(OptionContract* options, int n_opts, double spot, SVCJParams* out) {
    // Simplified Logic: Adjust implied parameters based on Put-Call skew
    // Real implementation requires FFT integration (Carr-Madan)
    
    double atm_vol = 0.0;
    int count = 0;
    
    for(int i=0; i<n_opts; i++) {
        // Extract ATM vol proxy
        if (fabs(options[i].strike - spot) / spot < 0.05) {
            // Invert Black-Scholes here (omitted for brevity)
            atm_vol += 0.2; // Dummy proxy
            count++;
        }
    }
    
    if (count > 0) atm_vol /= count;
    else atm_vol = 0.2;

    out->theta = atm_vol * atm_vol;
    out->v0 = out->theta;
    enforce_feller(out);
}