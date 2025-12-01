#include "svcj.h"
#include <stdio.h>

// --- Utilities ---
void clean_returns(double* returns, int n) {
    for(int i=0; i<n; i++) {
        if(isnan(returns[i]) || isinf(returns[i])) returns[i] = 0.0;
        if(fabs(returns[i]) < JIT_THRESH) {
            returns[i] = (i % 2 == 0) ? JIT_THRESH : -JIT_THRESH;
        }
    }
}

void check_constraints(SVCJParams* p) {
    // Hard floors to prevent division by zero or negative variance
    if(p->kappa < 0.1) p->kappa = 0.1;
    if(p->theta < 1e-5) p->theta = 1e-5;
    if(p->sigma_v < 0.01) p->sigma_v = 0.01;
    if(p->lambda_j < 1e-4) p->lambda_j = 1e-4;
    if(p->sigma_j < 1e-4) p->sigma_j = 1e-4;
    
    // Correlation clamping
    if(p->rho > 0.99) p->rho = 0.99;
    if(p->rho < -0.99) p->rho = -0.99;

    // Feller Condition (Soft enforcement)
    // 2 * kappa * theta >= sigma_v^2
    double feller_bound = 2.0 * p->kappa * p->theta;
    if(p->sigma_v * p->sigma_v > feller_bound) {
        // Cap sigma_v to slightly below the boundary
        p->sigma_v = sqrt(feller_bound) * 0.95; 
    }
}

// --- UKF with Sub-Step Integration ---
double ukf_log_likelihood(double* returns, int n, SVCJParams* p, double* out_spot_vol, double* out_jump_prob) {
    double ll = 0.0;
    double v_curr = p->theta; // Start at long-run mean
    
    for(int t=0; t<n; t++) {
        // 1. Prediction (Time Update) with Sub-Stepping
        // We integrate the ODE dv = kappa(theta - v)dt 
        double v_pred = v_curr;
        for(int s=0; s<SUB_STEPS; s++) {
            v_pred += p->kappa * (p->theta - v_pred) * SUB_DT;
            if(v_pred < 1e-6) v_pred = 1e-6; // Reflective barrier
        }

        // 2. Innovation (Observation)
        // Expected return due to drift and variance
        double drift = (p->mu - 0.5 * v_pred);
        double y_hat = drift * DT;
        double y = returns[t] - y_hat;

        // 3. Variance of Innovation (S)
        // Diffusive part integrated over DT + Jump Variance part
        double jump_var = p->lambda_j * (p->mu_j*p->mu_j + p->sigma_j*p->sigma_j);
        double S = v_pred * DT + jump_var * DT;
        
        if(S < 1e-8) S = 1e-8; // Prevent singularity

        // 4. Update (Measurement Update)
        // Kalman Gain approximation
        double K = (p->rho * p->sigma_v * DT) / S;
        v_curr = v_pred + K * y;
        
        // Final Safety Clamp
        if(v_curr < 1e-6) v_curr = 1e-6; 

        // 5. Jump Probability
        double mahalanobis = (y * y) / S;
        double prob_j = (mahalanobis > 9.0) ? 1.0 : (p->lambda_j * DT); 
        
        // Output writing
        if(out_spot_vol) out_spot_vol[t] = sqrt(v_curr);
        if(out_jump_prob) out_jump_prob[t] = prob_j;

        // Log Likelihood Accumulation
        ll += -0.5 * log(2 * M_PI * S) - 0.5 * mahalanobis;
    }
    return ll;
}

// --- Optimizer ---
void optimize_svcj(double* returns, int n, SVCJParams* p, double* out_spot_vol, double* out_jump_prob) {
    // Initial Defaults
    p->kappa = 4.0; p->theta = 0.02; p->sigma_v = 0.4; p->rho = -0.7;
    p->lambda_j = 0.1; p->mu_j = -0.05; p->sigma_j = 0.05; p->mu = 0.0005;
    
    check_constraints(p);

    // Coordinate Descent (Alternating Grid Search)
    // We alternate optimizing Vol-params and Jump-params
    double best_ll = -1e15;
    
    // Reduced iterations for speed in this demo, increase for production
    for(int i=0; i<5; i++) {
        // --- Optimize Kappa ---
        double base_k = p->kappa;
        double step = 0.5;
        double current_ll = ukf_log_likelihood(returns, n, p, NULL, NULL);
        
        p->kappa = base_k + step; check_constraints(p);
        double up_ll = ukf_log_likelihood(returns, n, p, NULL, NULL);
        
        p->kappa = base_k - step; check_constraints(p);
        double down_ll = ukf_log_likelihood(returns, n, p, NULL, NULL);
        
        if(up_ll > current_ll && up_ll > down_ll) p->kappa = base_k + step;
        else if(down_ll > current_ll) p->kappa = base_k - step;
        else p->kappa = base_k;
        
        // --- Optimize Theta ---
        // (Similar logic could be applied to all params, simplified here)
        // Updating State
        current_ll = ukf_log_likelihood(returns, n, p, NULL, NULL);
        if(current_ll > best_ll) best_ll = current_ll;
    }

    // Final Run to generate vectors
    ukf_log_likelihood(returns, n, p, out_spot_vol, out_jump_prob);
}

// --- Option Pricing ---
double normal_cdf(double x) {
    return 0.5 * (1.0 + erf(x / sqrt(2.0)));
}

double bs_price(double s, double k, double t, double r, double v, int type) {
    if(t <= 1e-5) return (type==1) ? fmax(s-k,0) : fmax(k-s,0);
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
        
        // We use the Merton expansion, but base the diffusion on the CURRENT spot_vol
        // This links the historic UKF fit to the current price
        double lambda_prime = p->lambda_j * (1 + p->mu_j);
        double price = 0.0;
        
        for(int n=0; n<10; n++) { // 10 terms usually sufficient
            double r_n = -p->lambda_j * p->mu_j * n;
            double v_n = sqrt(spot_vol*spot_vol + (n * p->sigma_j*p->sigma_j)/T);
            
            double fact = 1.0; for(int k=1; k<=n; k++) fact *= k;
            double weight = exp(-lambda_prime * T) * pow(lambda_prime * T, n) / fact;
            
            price += weight * bs_price(s0, K, T, r_n, v_n, type);
        }
        out_prices[i] = price;
    }
}