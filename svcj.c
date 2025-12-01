#include "svcj.h"
#include <stdio.h>

// --- Helper: Safe Copy & Denoise ---
// copies potentially non-contiguous or noisy data into a clean buffer
void clean_and_copy(double* src, double* dest, int n) {
    for(int i=0; i<n; i++) {
        double val = src[i];
        if(fabs(val) < JITTER_THRESHOLD) {
            val = (i % 2 == 0) ? JITTER_THRESHOLD : -JITTER_THRESHOLD;
        }
        dest[i] = val;
    }
}

// --- Helper: Parameter Constraints ---
void check_stability(SVCJParams* p) {
    if(p->kappa < 0.1) p->kappa = 0.1;
    if(p->theta < MIN_VAR) p->theta = MIN_VAR;
    if(p->sigma_v < 0.01) p->sigma_v = 0.01;
    
    // Feller Condition (Soft check, cap sigma_v if needed)
    double feller = 2.0 * p->kappa * p->theta;
    if(p->sigma_v*p->sigma_v > feller) {
        p->sigma_v = sqrt(feller * 0.99);
    }
    
    if(p->rho > 0.95) p->rho = 0.95;
    if(p->rho < -0.95) p->rho = -0.95;
    if(p->lambda_j < 0.001) p->lambda_j = 0.001;
}

// --- Core: UKF with Annualized State / Daily Step ---
double run_ukf_qmle(double* returns, int n, SVCJParams* p, double* out_spot_vol, double* out_jump_prob) {
    double log_likelihood = 0.0;
    
    // State: Annualized Variance (starts at long-run mean)
    double v_curr = p->theta; 
    
    for(int t=0; t<n; t++) {
        double ret_t = returns[t];

        // 1. Prediction (Euler Discretization of Annualized Process)
        // dv = kappa(theta - v)dt
        double v_pred = v_curr + p->kappa * (p->theta - v_curr) * DT;
        if(v_pred < MIN_VAR) v_pred = MIN_VAR;

        // 2. Innovation
        // Expected Return (Daily) ~ (mu - 0.5*v)*dt
        double expected_ret = (p->mu - 0.5 * v_pred) * DT;
        double y_tilde = ret_t - expected_ret;
        
        // 3. Variance of the Innovation
        // Var(ret) = v_pred * dt + jumps
        double jump_var_contrib = p->lambda_j * (p->mu_j*p->mu_j + p->sigma_j*p->sigma_j);
        double h_variance = (v_pred + jump_var_contrib) * DT;
        
        if(h_variance < 1e-9) h_variance = 1e-9;

        // 4. Update (Kalman Gain)
        // K = Cov(v, y) / Var(y)
        // Cov(v, y) approx rho * sigma_v * v * dt (simplified)
        // We use the correlation structure: K = rho * sigma_v * dt / dt = rho * sigma_v ?
        // Correct diffusion scaling: Cov ~ rho * sigma_v * sqrt(v) * sqrt(v) * dt ?
        // Standard Heston UKF Approximation:
        double kalman_gain = (p->rho * p->sigma_v * DT) / h_variance;
        
        // Clamp Gain for stability (Prevent over-reaction to single day spikes)
        if(kalman_gain > 50.0) kalman_gain = 50.0;
        if(kalman_gain < -50.0) kalman_gain = -50.0;
        
        v_curr = v_pred + kalman_gain * y_tilde;
        
        // Enforce Positivity
        if(v_curr < MIN_VAR) v_curr = MIN_VAR;
        if(v_curr > 2.0) v_curr = 2.0; // Cap at 200% Variance (approx 141% Vol)
        
        // 5. Jump Probability (Ex-Post)
        // Normalized innovation squared
        double z_score_sq = (y_tilde * y_tilde) / h_variance;
        double j_prob = (z_score_sq > 9.0) ? 1.0 : (p->lambda_j * DT); // 3-sigma event
        
        // Outputs
        // Return Annualized Volatility for consistent interpretation
        if(out_spot_vol) out_spot_vol[t] = sqrt(v_curr);
        if(out_jump_prob) out_jump_prob[t] = j_prob;

        log_likelihood += -0.5 * log(2 * M_PI * h_variance) - 0.5 * z_score_sq;
    }

    return -log_likelihood;
}

// --- Option Pricing ---
// Since parameters are now Annualized, no scaling is needed.
// Inputs: T is in Years.
void price_option_chain(double s0, double* strikes, double* expiries, int* types, int n_opts, SVCJParams* p, double* out_prices) {
    for(int i=0; i<n_opts; i++) {
        double K = strikes[i];
        double T = expiries[i]; 
        
        // Expected average variance over T
        // E[v_avg] = theta + (v0 - theta)*(1-e^-kT)/kT
        // Assuming v0 = theta for long-term pricing if not specified, or use p->theta
        double avg_var = p->theta; // Simplified for steady state
        
        double total_vol = sqrt(avg_var + p->lambda_j * (p->mu_j*p->mu_j + p->sigma_j*p->sigma_j));
        
        double d1 = (log(s0/K) + (p->mu + 0.5*total_vol*total_vol)*T) / (total_vol*sqrt(T));
        
        // Intrinsic + Time Value approximation
        double intrinsic = (types[i] == 1) ? (s0 - K) : (K - s0); 
        if(intrinsic < 0) intrinsic = 0;
        
        out_prices[i] = intrinsic + s0 * 0.15 * total_vol * sqrt(T); 
    }
}