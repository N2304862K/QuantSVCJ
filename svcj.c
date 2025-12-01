#include "svcj.h"
#include <stdio.h>

// --- Utilities ---

void clean_returns(double* returns, int n) {
    for(int i=0; i<n; i++) {
        // Zero-Return Jitter to prevent singularity in UKF covariance update
        if(fabs(returns[i]) < 1e-9) {
            returns[i] = (i % 2 == 0) ? 1e-9 : -1e-9;
        }
    }
}

void check_constraints(SVCJParams* p) {
    if(p->kappa < 0.1) p->kappa = 0.1;
    if(p->theta < 1e-5) p->theta = 1e-5;
    if(p->sigma_v < 0.01) p->sigma_v = 0.01;
    if(p->lambda_j < 0.001) p->lambda_j = 0.001;
    if(p->sigma_j < 1e-4) p->sigma_j = 1e-4;
    
    // Correlation Bound
    if(p->rho > 0.95) p->rho = 0.95;
    if(p->rho < -0.95) p->rho = -0.95;

    // Feller Condition Enforcer (Soft boundary)
    if(2 * p->kappa * p->theta < p->sigma_v * p->sigma_v) {
        p->sigma_v = sqrt(2 * p->kappa * p->theta) * 0.99;
    }
}

// --- Unscented Kalman Filter (UKF) ---

double ukf_log_likelihood(double* returns, int n, SVCJParams* p, double* out_spot_vol, double* out_jump_prob) {
    double ll = 0.0;
    double v = p->theta; // State: Variance
    
    for(int t=0; t<n; t++) {
        // 1. Predict
        double v_pred = v + p->kappa * (p->theta - v) * DT;
        if(v_pred < 1e-6) v_pred = 1e-6;

        // 2. Innovation
        double drift = (p->mu - 0.5 * v_pred);
        double y_hat = drift * DT;
        double y = returns[t] - y_hat;

        // 3. Variance of Innovation (Diffusive + Jump)
        double jump_var = p->lambda_j * (p->mu_j*p->mu_j + p->sigma_j*p->sigma_j);
        double S = v_pred * DT + jump_var * DT;
        
        // 4. Update (Kalman Gain)
        double K = (p->rho * p->sigma_v * DT) / S;
        v = v_pred + K * y;
        
        // 5. Jump Probability (Instantaneous)
        // Mahalanobis distance based outlier detection
        double mahalanobis = (y * y) / S;
        double prob_j = (mahalanobis > 4.0) ? 1.0 : (p->lambda_j * DT);
        
        if(out_spot_vol) out_spot_vol[t] = sqrt(v);
        if(out_jump_prob) out_jump_prob[t] = prob_j;

        ll += -0.5 * log(2 * M_PI * S) - 0.5 * (y * y) / S;
    }
    return ll;
}

// --- Internal Optimizer (Coordinate Descent) ---
void optimize_svcj(double* returns, int n, SVCJParams* p, double* out_spot_vol, double* out_jump_prob) {
    double best_ll = -1e9;
    
    // Initial guess
    p->kappa = 2.0; p->theta = 0.04; p->sigma_v = 0.3; p->rho = -0.6;
    p->lambda_j = 0.1; p->mu_j = 0.0; p->sigma_j = 0.05; p->mu = 0.0;
    check_constraints(p);

    // Simple Coordinate Descent for Robustness
    // In production, this would be BFGS, but for a standalone C-file, this is chemically stable.
    double steps[] = {0.05, 0.01};
    
    for(int iter=0; iter<10; iter++) { // 10 passes
        // Optimize Kappa
        double curr_ll = ukf_log_likelihood(returns, n, p, NULL, NULL);
        double old_val = p->kappa;
        p->kappa += steps[0]; check_constraints(p);
        if(ukf_log_likelihood(returns, n, p, NULL, NULL) > curr_ll) continue;
        p->kappa = old_val - steps[0]; check_constraints(p);
        if(ukf_log_likelihood(returns, n, p, NULL, NULL) > curr_ll) continue;
        p->kappa = old_val; // Revert
    }
    
    // Final Pass to populate outputs
    ukf_log_likelihood(returns, n, p, out_spot_vol, out_jump_prob);
}

// --- Option Pricing (Merton Jump Diffusion Expansion) ---
// Uses the Spot Volatility from UKF as the baseline diffusion
double normal_cdf(double x) {
    return 0.5 * (1.0 + erf(x / sqrt(2.0)));
}

double bs_price(double s, double k, double t, double r, double v, int type) {
    if(t <= 0) return (type==1) ? fmax(s-k,0) : fmax(k-s,0);
    double d1 = (log(s/k) + (r + 0.5*v*v)*t) / (v*sqrt(t));
    double d2 = d1 - v*sqrt(t);
    if(type == 1) return s * normal_cdf(d1) - k * exp(-r*t) * normal_cdf(d2);
    else return k * exp(-r*t) * normal_cdf(-d2) - s * normal_cdf(-d1);
}

void price_option_chain(double s0, double* strikes, double* expiries, int* types, int n_opts, SVCJParams* p, double spot_vol, double* out_prices) {
    for(int i=0; i<n_opts; i++) {
        double K = strikes[i];
        double T = expiries[i];
        int type = types[i];
        
        // Merton Series Expansion for Jumps
        // Sum of weighted BS prices
        double price_sum = 0.0;
        double lambda_prime = p->lambda_j * (1 + p->mu_j);
        
        for(int k=0; k<MERTON_ITER; k++) {
            double r_k = -p->lambda_j * p->mu_j * k; // Simplified risk neutral drift adj
            double vol_k = sqrt(spot_vol*spot_vol + (k * p->sigma_j * p->sigma_j) / T);
            
            // Poisson weight
            double fact = 1.0; for(int j=1; j<=k; j++) fact *= j;
            double weight = exp(-lambda_prime * T) * pow(lambda_prime * T, k) / fact;
            
            price_sum += weight * bs_price(s0, K, T, r_k, vol_k, type);
        }
        out_prices[i] = price_sum;
    }
}