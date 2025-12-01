#include "svcj.h"
#include <float.h>

// --- Utilities ---

void clean_returns(double* returns, int n) {
    for(int i=0; i<n; i++) {
        // Prevent absolute zero which can cause singularity in variance updates
        if(fabs(returns[i]) < JITTER) {
            returns[i] = (i % 2 == 0) ? JITTER : -JITTER;
        }
    }
}

void check_constraints(SVCJParams* p) {
    // Enforce domain constraints for stable estimation
    if(p->kappa < 0.1) p->kappa = 0.1;
    if(p->kappa > 20.0) p->kappa = 20.0;
    
    if(p->theta < 0.001) p->theta = 0.001; // Min var 0.1%
    if(p->theta > 2.0) p->theta = 2.0;

    if(p->sigma_v < 0.01) p->sigma_v = 0.01;
    if(p->sigma_v > 2.0) p->sigma_v = 2.0;

    if(p->lambda_j < 0.01) p->lambda_j = 0.01;
    if(p->lambda_j > 100.0) p->lambda_j = 100.0; // Cap jumps per year

    if(p->sigma_j < 0.001) p->sigma_j = 0.001;
    
    // Correlation Bound
    if(p->rho > 0.95) p->rho = 0.95;
    if(p->rho < -0.95) p->rho = -0.95;

    // Feller Condition Enforcer: 2*kappa*theta >= sigma_v^2
    // We enforce a soft boundary to keep square root valid
    double feller_bound = 2.0 * p->kappa * p->theta;
    if(p->sigma_v * p->sigma_v > feller_bound) {
        p->sigma_v = sqrt(feller_bound) * 0.95; 
    }
}

// --- Unscented Kalman Filter (UKF) ---

double ukf_log_likelihood(double* returns, int n, SVCJParams* p, double* out_spot_vol, double* out_jump_prob) {
    double ll = 0.0;
    double v = p->theta; // Initial State: Long run variance
    
    for(int t=0; t<n; t++) {
        // 1. Predict
        double v_pred = v + p->kappa * (p->theta - v) * DT;
        if(v_pred < 1e-6) v_pred = 1e-6; // prevent negative variance

        // 2. Innovation
        // Expected return approx (mu - 0.5*v) * dt
        double drift = (p->mu - 0.5 * v_pred);
        double y_hat = drift * DT;
        double y = returns[t] - y_hat;

        // 3. Variance of Innovation S
        // Diffusive part + Jump part variance expectation
        double jump_var_contrib = p->lambda_j * (p->mu_j*p->mu_j + p->sigma_j*p->sigma_j);
        double S = v_pred * DT + jump_var_contrib * DT;
        
        if(S < 1e-8) S = 1e-8; // Numerical safety
        if(isnan(S) || isinf(S)) return -1e9; // Fail gracefully

        // 4. Update (Kalman Gain)
        // Covariance(State, Measurement) approx rho * sigma_v * v * dt ?
        // Simplified Linear QMLE approx:
        double K = (p->rho * p->sigma_v * DT) / S;
        
        // State Update
        v = v_pred + K * y;
        
        // Enforce positivity on updated state
        if(v < 1e-6) v = 1e-6;

        // 5. Jump Probability (Instantaneous)
        // Using Mahalanobis distance
        double mahalanobis = (y * y) / S;
        double prob_j = 0.0;
        
        // Heuristic: If move is > 3 std devs of diffusive vol, high jump prob
        if (mahalanobis > 9.0) prob_j = 0.99;
        else prob_j = p->lambda_j * DT; 
        
        if(out_spot_vol) out_spot_vol[t] = sqrt(v); // Return Vol (sqrt(var))
        if(out_jump_prob) out_jump_prob[t] = prob_j;

        // Log Likelihood Accumulation
        double step_ll = -0.5 * log(2 * M_PI * S) - 0.5 * (y * y) / S;
        if(isnan(step_ll) || isinf(step_ll)) return -1e9;
        ll += step_ll;
    }
    return ll;
}

// --- Internal Optimizer (Coordinate Descent) ---
void optimize_svcj(double* returns, int n, SVCJParams* p, double* out_spot_vol, double* out_jump_prob) {
    // Defaults
    p->mu = 0.05; p->kappa = 2.0; p->theta = 0.04; p->sigma_v = 0.2; 
    p->rho = -0.5; p->lambda_j = 5.0; p->mu_j = -0.05; p->sigma_j = 0.05;
    
    check_constraints(p);
    
    double best_ll = ukf_log_likelihood(returns, n, p, NULL, NULL);
    if(best_ll <= -1e8) { 
        // Logic failure on defaults? Reset to safer defaults
        p->theta = 0.02; p->sigma_v = 0.1;
        best_ll = ukf_log_likelihood(returns, n, p, NULL, NULL);
    }

    // Coordinate Descent config
    double step_size = 0.05;
    int max_iter = 15; // Kept low for speed in this demo
    
    // Params to optimize: kappa, theta, sigma_v, rho
    // We hold jumps constant for stability in this simple version unless extended
    for(int i=0; i<max_iter; i++) {
        int improved = 0;
        
        // Optimize Kappa
        double old_k = p->kappa;
        p->kappa += step_size; check_constraints(p);
        double ll = ukf_log_likelihood(returns, n, p, NULL, NULL);
        if(ll > best_ll) { best_ll = ll; improved=1; }
        else {
            p->kappa = old_k - step_size; check_constraints(p);
            ll = ukf_log_likelihood(returns, n, p, NULL, NULL);
            if(ll > best_ll) { best_ll = ll; improved=1; }
            else p->kappa = old_k;
        }

        // Optimize Theta
        double old_t = p->theta;
        p->theta += step_size * 0.01; check_constraints(p);
        ll = ukf_log_likelihood(returns, n, p, NULL, NULL);
        if(ll > best_ll) { best_ll = ll; improved=1; }
        else {
            p->theta = old_t - step_size * 0.01; check_constraints(p);
            ll = ukf_log_likelihood(returns, n, p, NULL, NULL);
            if(ll > best_ll) { best_ll = ll; improved=1; }
            else p->theta = old_t;
        }
        
        if(!improved && step_size > 0.001) step_size *= 0.5; // Decay step
    }

    // Final Run to populate outputs
    ukf_log_likelihood(returns, n, p, out_spot_vol, out_jump_prob);
}

// --- Option Pricing ---
// Standard Gaussian CDF
double norm_cdf(double x) {
    return 0.5 * erfc(-x * M_SQRT1_2);
}

double bs_formula(double S, double K, double T, double r, double sigma, int type) {
    if (T <= 1e-4) return (type == 1) ? fmax(S - K, 0.0) : fmax(K - S, 0.0);
    double d1 = (log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * sqrt(T));
    double d2 = d1 - sigma * sqrt(T);
    if (type == 1) return S * norm_cdf(d1) - K * exp(-r * T) * norm_cdf(d2);
    else return K * exp(-r * T) * norm_cdf(-d2) - S * norm_cdf(-d1);
}

void price_option_chain(double s0, double* strikes, double* expiries, int* types, int n_opts, SVCJParams* p, double spot_vol, double* out_prices) {
    // Merton Jump Diffusion Approximation
    // We treat 'spot_vol' as the diffusive component sigma
    
    double lambda = p->lambda_j;
    double m = p->mu_j;
    double v = p->sigma_j;
    double r = p->mu; // Use drift as risk-free proxy for this context
    
    // Pre-calc Lambda prime
    double lambda_p = lambda * (1.0 + m);
    
    for(int i=0; i<n_opts; i++) {
        double K = strikes[i];
        double T = expiries[i];
        int type = types[i];
        
        double weighted_price = 0.0;
        
        // Sum first 10 terms of Poisson expansion
        for(int k=0; k<10; k++) {
            double factorial = 1.0;
            for(int j=1; j<=k; j++) factorial *= j;
            
            // Poisson probability of k jumps
            double prob_k = exp(-lambda_p * T) * pow(lambda_p * T, k) / factorial;
            
            if(prob_k < 1e-6 && k > 2) break; // Optimization
            
            // Adjusted parameters for the k-jump conditioned BS
            double r_k = r - lambda * m + (k * log(1.0 + m)) / T;
            double sigma_k = sqrt(spot_vol*spot_vol + (k * v*v) / T);
            
            weighted_price += prob_k * bs_formula(s0, K, T, r_k, sigma_k, type);
        }
        
        out_prices[i] = weighted_price;
    }
}