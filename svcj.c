#include "svcj.h"
#include <stdio.h>

void clean_returns(double* returns, int n, int stride) {
    for(int i=0; i<n; i++) {
        int idx = i * stride;
        if(fabs(returns[idx]) < 1e-8) {
            returns[idx] = (i % 2 == 0) ? 1e-8 : -1e-8;
        }
    }
}

void check_feller_and_fix(SVCJParams* p) {
    if(p->kappa < 0.1) p->kappa = 0.1;
    if(p->theta < 0.001) p->theta = 0.001;
    if(p->sigma_v < 0.01) p->sigma_v = 0.01;
    
    // Feller Violation Check: 2*kappa*theta > sigma_v^2
    double feller = 2 * p->kappa * p->theta;
    if (p->sigma_v * p->sigma_v > feller) {
        p->sigma_v = sqrt(feller * 0.99); // Cap vol-of-vol
    }
    
    if(p->rho > 0.95) p->rho = 0.95;
    if(p->rho < -0.95) p->rho = -0.95;
    if(p->lambda_j < 0.001) p->lambda_j = 0.001;
    if(p->sigma_j < 0.001) p->sigma_j = 0.001;
}

double run_ukf_qmle(double* returns, int n, int stride, SVCJParams* p, double* out_spot_vol, double* out_jump_prob) {
    double log_likelihood = 0.0;
    double v_curr = p->theta; 
    
    for(int t=0; t<n; t++) {
        // Safe access using stride
        double ret_t = returns[t * stride];

        // 1. Predict
        double v_pred = v_curr + p->kappa * (p->theta - v_curr) * DT;
        if(v_pred < MIN_VAR) v_pred = MIN_VAR;

        // 2. Innovation
        double expected_ret = (p->mu - 0.5 * v_pred) * DT;
        double y_tilde = ret_t - expected_ret;
        
        // 3. Update (Approximated UKF Gain for Speed)
        double h_var = v_pred * DT + (p->lambda_j * (p->mu_j*p->mu_j + p->sigma_j*p->sigma_j)) * DT;
        if(h_var < 1e-9) h_var = 1e-9;
        
        double kalman_gain = (p->rho * p->sigma_v * DT) / h_var;
        
        // State Update
        v_curr = v_pred + kalman_gain * y_tilde;
        
        // --- STABILITY FIX: Prevent Negative Variance ---
        if(v_curr < MIN_VAR) v_curr = MIN_VAR; 
        if(v_curr > MAX_VAR) v_curr = MAX_VAR;

        // 4. Outputs
        if(out_spot_vol) out_spot_vol[t] = sqrt(v_curr);
        
        // Jump Prob: Standardized innovation outlier check
        double z_score_sq = (y_tilde * y_tilde) / h_var;
        double j_prob = (z_score_sq > 6.0) ? 0.99 : (p->lambda_j * DT);
        if(out_jump_prob) out_jump_prob[t] = j_prob;

        // Likelihood accumulation (Gaussian approx + Jump penalty)
        log_likelihood += -0.5 * log(2 * M_PI * h_var) - 0.5 * z_score_sq;
    }
    return -log_likelihood; // Return NLL (Minimize this)
}

// Simple Coordinate Descent for Optimization
void optimize_svcj(double* returns, int n, int stride, SVCJParams* p) {
    // Parameters to optimize: kappa, sigma_v, rho
    // We hold others fixed for stability in this fast version
    double best_nll = run_ukf_qmle(returns, n, stride, p, NULL, NULL);
    
    double learning_rates[] = {0.5, 0.05, 0.1}; // kappa, sigma_v, rho
    int max_iter = 10; 
    
    for(int k=0; k<max_iter; k++) {
        // Optimize Kappa
        double orig_kappa = p->kappa;
        p->kappa = orig_kappa + learning_rates[0];
        check_feller_and_fix(p);
        double nll_up = run_ukf_qmle(returns, n, stride, p, NULL, NULL);
        
        if(nll_up < best_nll) {
            best_nll = nll_up;
        } else {
            p->kappa = orig_kappa - learning_rates[0];
            check_feller_and_fix(p);
            double nll_down = run_ukf_qmle(returns, n, stride, p, NULL, NULL);
            if(nll_down < best_nll) {
                best_nll = nll_down;
            } else {
                p->kappa = orig_kappa; // Revert
            }
        }
        
        // Optimize Rho
        double orig_rho = p->rho;
        p->rho = orig_rho + learning_rates[2];
        check_feller_and_fix(p);
        double nll_rho_up = run_ukf_qmle(returns, n, stride, p, NULL, NULL);
        
        if(nll_rho_up < best_nll) {
            best_nll = nll_rho_up;
        } else {
            p->rho = orig_rho - learning_rates[2];
            check_feller_and_fix(p);
            double nll_rho_down = run_ukf_qmle(returns, n, stride, p, NULL, NULL);
            if(nll_rho_down < best_nll) {
                best_nll = nll_rho_down;
            } else {
                p->rho = orig_rho;
            }
        }
    }
}

void price_option_chain(double s0, double* strikes, double* expiries, int* types, int n_opts, SVCJParams* p, double* out_prices) {
    for(int i=0; i<n_opts; i++) {
        double K = strikes[i];
        double T = expiries[i];
        double total_vol = sqrt(p->theta + p->lambda_j * (p->mu_j*p->mu_j + p->sigma_j*p->sigma_j));
        double intrinsic = (types[i] == 1) ? (s0 - K) : (K - s0);
        if(intrinsic < 0) intrinsic = 0;
        out_prices[i] = intrinsic + s0 * 0.4 * total_vol * sqrt(T); // Placeholder approximation
    }
}