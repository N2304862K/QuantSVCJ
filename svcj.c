#include "svcj.h"

// --- Utilities ---
double sq(double x) { return x * x; }

void clean_returns(double* returns, int n, int stride) {
    for(int i=0; i<n; i++) {
        if(fabs(returns[i*stride]) < JITTER) {
            returns[i*stride] = (i%2==0) ? JITTER : -JITTER;
        }
    }
}

void constrain_params(SVCJParams* p) {
    // Safety Clamps for Stability
    if(p->theta < 1e-6) p->theta = 1e-6;      // Min Daily Variance
    if(p->kappa < 1e-3) p->kappa = 1e-3;      // Min Mean Reversion
    if(p->sigma_v < 1e-4) p->sigma_v = 1e-4;  // Min Vol of Vol
    
    // Feller Condition Check (2*kappa*theta > sigma_v^2)
    double feller = 2.0 * p->kappa * p->theta;
    if(p->sigma_v * p->sigma_v > feller) {
        p->sigma_v = sqrt(feller * 0.99);
    }
    
    // Correlation
    if(p->rho > 0.99) p->rho = 0.99;
    if(p->rho < -0.99) p->rho = -0.99;
    
    // Jumps
    if(p->lambda_j < 0.0001) p->lambda_j = 0.0001; // Min 1 jump every 40 years approx
    if(p->lambda_j > 0.5) p->lambda_j = 0.5;       // Max jump freq (every 2 days)
    if(p->sigma_j < 0.001) p->sigma_j = 0.001;
}

// --- 1. QMLE Core (The Filter) ---
// Returns: Negative Log Likelihood (Lower is Better)
double run_ukf_likelihood(double* returns, int n, int stride, SVCJParams* p, double* out_spot, double* out_jump) {
    double nll = 0.0;
    double v = p->theta; // Start at long run mean
    
    for(int t=0; t<n; t++) {
        double ret = returns[t*stride];
        
        // 1. Predict
        double v_pred = v + p->kappa * (p->theta - v) * DT;
        if(v_pred < 1e-7) v_pred = 1e-7;
        
        // 2. Innovation
        // Total Variance = Continuous + Jump Variance
        double jump_var = p->lambda_j * (sq(p->mu_j) + sq(p->sigma_j));
        double h_t = v_pred + jump_var; 
        
        double expected_ret = (p->mu - 0.5 * h_t) * DT;
        double y = ret - expected_ret;
        
        // 3. Update (Simple UKF Gain)
        // If variance is tiny, gain explodes, so clamp h_t
        if(h_t < 1e-7) h_t = 1e-7;
        
        double K = (p->rho * p->sigma_v * DT) / h_t;
        v = v_pred + K * y;
        
        // Clamp updated variance
        if(v < 1e-7) v = 1e-7;
        
        // 4. Record Stats
        if(out_spot) out_spot[t*stride] = sqrt(v); // Store Vol, not Var
        if(out_jump) {
            // Instantaneous Jump Prob via Mahalanobis distance
            double dist = (y*y) / h_t;
            // Sigmoid-like activation for jump probability
            out_jump[t*stride] = (dist > 9.0) ? 1.0 : (p->lambda_j * DT); 
        }
        
        // 5. Accumulate NLL
        nll += 0.5 * log(2 * M_PI * h_t) + 0.5 * (y*y) / h_t;
    }
    return nll;
}

// --- 2. Optimizer: Coordinate Descent for History ---
void calibrate_to_history(double* returns, int n, int stride, SVCJParams* p) {
    double best_nll = run_ukf_likelihood(returns, n, stride, p, NULL, NULL);
    double step = 0.1; // 10% perturbation
    
    for(int i=0; i<MAX_OPT_ITER; i++) {
        int improved = 0;
        
        // Parameters to optimize: Kappa, Theta, Sigma_V, Rho
        // We use pointers to iterate over struct members? No, explicit is safer in C.
        
        double* targets[] = {&(p->kappa), &(p->theta), &(p->sigma_v), &(p->rho)};
        double orig_vals[4];
        
        for(int k=0; k<4; k++) {
            orig_vals[k] = *targets[k];
            
            // Try increasing
            *targets[k] = orig_vals[k] * (1.0 + step);
            constrain_params(p);
            double nll_up = run_ukf_likelihood(returns, n, stride, p, NULL, NULL);
            
            if(nll_up < best_nll) {
                best_nll = nll_up;
                improved = 1;
                continue;
            }
            
            // Try decreasing
            *targets[k] = orig_vals[k] * (1.0 - step);
            constrain_params(p);
            double nll_down = run_ukf_likelihood(returns, n, stride, p, NULL, NULL);
            
            if(nll_down < best_nll) {
                best_nll = nll_down;
                improved = 1;
                continue;
            }
            
            // Revert if no improvement
            *targets[k] = orig_vals[k];
        }
        
        if(!improved) step *= 0.5; // Refine search
        if(step < 0.001) break;    // Converged
    }
}

// --- 3. Option Pricing (BSM with Annualized SVCJ inputs) ---
void price_option_chain(double s0, double* strikes, double* expiries, int* types, int n_opts, SVCJParams* p, double* out_prices) {
    for(int i=0; i<n_opts; i++) {
        double K = strikes[i];
        double T = expiries[i]; // Years
        
        // CONVERSION: Daily -> Annual
        double theta_a = p->theta * DAYS_PER_YEAR;
        double kappa_a = p->kappa * DAYS_PER_YEAR; // Mean rev speed scales with time? Yes, roughly.
        double lambda_a = p->lambda_j * DAYS_PER_YEAR;
        
        // Expected Average Variance over T
        // E[v] = theta + (v0 - theta)*(...) -> Assuming starts at long run theta
        double exp_var = theta_a; 
        double jump_comp = lambda_a * (sq(p->mu_j) + sq(p->sigma_j));
        double total_vol = sqrt(exp_var + jump_comp);
        
        if(total_vol < 0.01) total_vol = 0.01;

        double d1 = (log(s0/K) + (0.5*total_vol*total_vol)*T) / (total_vol*sqrt(T));
        double d2 = d1 - total_vol*sqrt(T);
        
        // Simple BSM approximation using Total Volatility
        // (Full FFT is too complex for this snippet size constraint)
        // Standard Normal CDF approx
        double nd1 = 0.5 * erfc(-d1 * M_SQRT1_2);
        double nd2 = 0.5 * erfc(-d2 * M_SQRT1_2);
        
        if(types[i] == 1) { // Call
            out_prices[i] = s0 * nd1 - K * exp(0) * nd2; // r=0 assumption for simplicity
        } else { // Put
            out_prices[i] = K * exp(0) * (1 - nd2) - s0 * (1 - nd1);
        }
    }
}

// --- 4. Optimizer: Calibration to Options (Fit Jumps) ---
void calibrate_to_options(double s0, double* strikes, double* expiries, int* types, double* mkt_prices, int n_opts, SVCJParams* p) {
    // We assume Kappa/Theta/Rho are fixed by history. 
    // We tweak Lambda, Mu_J, Sigma_J to fit smile.
    
    double *prices = (double*)malloc(n_opts * sizeof(double));
    double best_mse = 1e9;
    
    // Initial Calc
    price_option_chain(s0, strikes, expiries, types, n_opts, p, prices);
    for(int i=0; i<n_opts; i++) best_mse += sq(prices[i] - mkt_prices[i]);
    
    double step = 0.2;
    
    for(int iter=0; iter<20; iter++) {
        double* targets[] = {&(p->lambda_j), &(p->mu_j), &(p->sigma_j)};
        int improved = 0;
        
        for(int k=0; k<3; k++) {
            double orig = *targets[k];
            
            // Up
            *targets[k] = orig * (1.0 + step);
            constrain_params(p);
            
            double mse = 0;
            price_option_chain(s0, strikes, expiries, types, n_opts, p, prices);
            for(int j=0; j<n_opts; j++) mse += sq(prices[j] - mkt_prices[j]);
            
            if(mse < best_mse) {
                best_mse = mse;
                improved = 1;
                continue;
            }
            
            // Down
            *targets[k] = orig * (1.0 - step);
            constrain_params(p);
            
            mse = 0;
            price_option_chain(s0, strikes, expiries, types, n_opts, p, prices);
            for(int j=0; j<n_opts; j++) mse += sq(prices[j] - mkt_prices[j]);
            
            if(mse < best_mse) {
                best_mse = mse;
                improved = 1;
                continue;
            }
            
            *targets[k] = orig;
        }
        if(!improved) step *= 0.5;
    }
    free(prices);
}