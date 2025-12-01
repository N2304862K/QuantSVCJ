#include "svcj.h"

// --- Utils: Micro-noise to prevent singular matrices ---
void clean_returns(double* returns, int n) {
    for(int i=0; i<n; i++) {
        if(fabs(returns[i]) < JITTER_THRESHOLD) {
            returns[i] = (i % 2 == 0) ? JITTER_THRESHOLD : -JITTER_THRESHOLD;
        }
    }
}

// --- Utils: Constraint Enforcement ---
void check_feller_and_fix(SVCJParams* p) {
    // 1. Boundaries
    if(p->kappa < 0.1) p->kappa = 0.1;
    if(p->theta < 1e-5) p->theta = 1e-5;
    if(p->sigma_v < 0.05) p->sigma_v = 0.05;
    if(p->lambda_j < 1e-4) p->lambda_j = 1e-4;
    if(p->sigma_j < 1e-4) p->sigma_j = 1e-4;
    
    // 2. Correlation Clamp
    if(p->rho > 0.95) p->rho = 0.95;
    if(p->rho < -0.95) p->rho = -0.95;

    // 3. Feller Condition (Soft enforcement)
    // 2 * kappa * theta > sigma_v^2
    double feller = 2.0 * p->kappa * p->theta;
    if (p->sigma_v * p->sigma_v > feller) {
        p->sigma_v = sqrt(feller) * 0.99; 
    }
}

// --- Utils: Bootstrap Variance ---
double estimate_initial_variance(double* returns, int n) {
    int lookback = (n < 20) ? n : 20;
    double sum_sq = 0.0;
    for(int i=0; i<lookback; i++) {
        sum_sq += returns[i] * returns[i];
    }
    double res = sum_sq / lookback;
    return (res < 1e-5) ? 1e-5 : res; // Annualization happens in usage
}

// --- Core: UKF / QMLE Step ---
// Returns: Negative Log Likelihood (Minimization Target)
double run_ukf_qmle(double* returns, int n, SVCJParams* p, double* out_spot_vol, double* out_jump_prob) {
    double nll = 0.0;
    double v_curr = p->theta; // Start at long run mean
    
    // Attempt to seed better if not optimizing
    if(out_spot_vol != NULL) {
         v_curr = estimate_initial_variance(returns, n) / DT;
    }

    for(int t=0; t<n; t++) {
        // 1. SVCJ Euler Discretization
        double v_pred = v_curr + p->kappa * (p->theta - v_curr) * DT;
        if(v_pred < 1e-6) v_pred = 1e-6; // Physical barrier

        // 2. Expected Return & Innovation
        // Drift + Risk Premium adjustment
        double expected_ret = (p->mu - 0.5 * v_pred) * DT; 
        double y_tilde = returns[t] - expected_ret;

        // 3. Variance of Innovation (H)
        // Includes diffusive var and expected jump var
        double jump_var_contrib = p->lambda_j * (p->mu_j*p->mu_j + p->sigma_j*p->sigma_j);
        double h_var = v_pred * DT + jump_var_contrib * DT;
        
        if(h_var < 1e-8 || isnan(h_var)) {
            nll += PENALTY_VAL; // Penalty for numerical failure
            v_curr = p->theta; // Reset
            continue;
        }

        // 4. Update Step (Simple UKF Gain)
        double k_gain = (p->rho * p->sigma_v * DT) / h_var;
        v_curr = v_pred + k_gain * y_tilde;
        
        // Clamp updated variance
        if(v_curr < 1e-6) v_curr = 1e-6;
        if(v_curr > 5.0) v_curr = 5.0; // Cap explosion

        // 5. Likelihood Accumulation
        double step_ll = -0.5 * log(2 * M_PI * h_var) - 0.5 * (y_tilde * y_tilde) / h_var;
        nll -= step_ll; // We want to minimize Negative LL

        // 6. Output Generation
        if(out_spot_vol) out_spot_vol[t] = sqrt(v_curr);
        
        if(out_jump_prob) {
            // Simplified Jump detection based on standardized innovation
            double z_score = (y_tilde * y_tilde) / h_var;
            // Sigmoid-like probability
            double prob = 1.0 / (1.0 + exp(-(z_score - 4.0))); 
            out_jump_prob[t] = prob; 
        }
    }
    
    if(isnan(nll) || isinf(nll)) return PENALTY_VAL;
    return nll;
}

// --- Core: Coordinate Descent Optimizer ---
void fit_svcj_history(double* returns, int n, SVCJParams* p) {
    double best_nll = run_ukf_qmle(returns, n, p, NULL, NULL);
    double step_sizes[] = {0.5, 0.01, 0.1, 0.05, 0.05, 0.01}; // kappa, theta, sig_v, rho, lam, mu_j
    
    // Simple Coordinate Descent
    for(int iter=0; iter<MAX_ITER; iter++) {
        int improved = 0;
        
        // Loop over params (simplification: only optimizing key params)
        // 0: kappa, 1: theta, 2: sigma_v, 3: rho
        double* targets[] = {&p->kappa, &p->theta, &p->sigma_v, &p->rho};
        
        for(int i=0; i<4; i++) {
            double original = *targets[i];
            double step = step_sizes[i];
            
            // Try +step
            *targets[i] = original + step;
            check_feller_and_fix(p);
            double nll_up = run_ukf_qmle(returns, n, p, NULL, NULL);
            
            if(nll_up < best_nll) {
                best_nll = nll_up;
                improved = 1;
                continue; // Keep change
            }
            
            // Try -step
            *targets[i] = original - step;
            check_feller_and_fix(p);
            double nll_down = run_ukf_qmle(returns, n, p, NULL, NULL);
            
            if(nll_down < best_nll) {
                best_nll = nll_down;
                improved = 1;
                continue; // Keep change
            }
            
            // Revert
            *targets[i] = original;
        }
        
        if(!improved) break; // Convergence
    }
}

// --- Option Pricing & Calibration ---
void calibrate_option_adjustment(double s0, double* strikes, double* expiries, int* types, double* mkt_prices, int n_opts, SVCJParams* p) {
    // 1. First, fit history to get baseline "P" measure parameters
    // (This assumes params struct is already populated/fit by wrapper via fit_svcj_history)

    // 2. Adjust Risk Premia (Lambda / Theta) to minimize pricing error
    // Simple Grid Search for Implied Params
    double best_err = 1e15;
    double best_theta = p->theta;
    double original_theta = p->theta;
    
    // Perturb Theta (Risk Neutral Variance) to fit options
    for(double mult = 0.5; mult <= 2.0; mult += 0.1) {
        p->theta = original_theta * mult;
        double current_err = 0.0;
        
        for(int i=0; i<n_opts; i++) {
            double K = strikes[i];
            double T = expiries[i];
            if(T < 0.001) T = 0.001; // Safety
            
            // Analytic Approx for SVCJ Option Price (BSM with Adjusted Vol)
            // Var[T] approx = theta + (v0 - theta)...
            double exp_var = p->theta; // Long run approx
            double tot_vol = sqrt(exp_var + p->lambda_j * (p->mu_j*p->mu_j + p->sigma_j*p->sigma_j));
            
            double d1 = (log(s0/K) + (0.0 + 0.5*tot_vol*tot_vol)*T) / (tot_vol*sqrt(T));
            double d2 = d1 - tot_vol*sqrt(T);
            
            // Normal CDF approx
            double nd1 = 0.5 * erfc(-d1 * M_SQRT1_2);
            double nd2 = 0.5 * erfc(-d2 * M_SQRT1_2);
            
            double model_price;
            if(types[i] == 1) model_price = s0 * nd1 - K * exp(0.0) * nd2; // Call (r=0 for now)
            else              model_price = K * exp(0.0) * (1-nd2) - s0 * (1-nd1); // Put
            
            current_err += (model_price - mkt_prices[i]) * (model_price - mkt_prices[i]);
        }
        
        if(current_err < best_err) {
            best_err = current_err;
            best_theta = p->theta;
        }
    }
    
    p->theta = best_theta; // Set to best fit
}