#include "svcj.h"
#include <float.h>

// --- Utilities ---

void clean_returns(double* returns, int n) {
    for(int i=0; i<n; i++) {
        // Inject microscopic noise into absolute zeros to prevent singular matrices
        if(fabs(returns[i]) < 1e-12) {
            returns[i] = (i % 2 == 0) ? 1e-12 : -1e-12;
        }
    }
}

void check_constraints(SVCJParams* p) {
    // 1. Mean Reversion (Kappa): Must be positive, usually < 20
    if(p->kappa < 0.1) p->kappa = 0.1;
    if(p->kappa > 30.0) p->kappa = 30.0;

    // 2. Long Run Variance (Theta): Positive, reasonable cap (2000% vol)
    if(p->theta < 1e-5) p->theta = 1e-5;
    if(p->theta > 0.5) p->theta = 0.5;

    // 3. Vol of Vol (Sigma_v): Positive
    if(p->sigma_v < 0.01) p->sigma_v = 0.01;
    if(p->sigma_v > 4.0) p->sigma_v = 4.0;

    // 4. Correlation (Rho): [-1, 1]
    if(p->rho > 0.99) p->rho = 0.99;
    if(p->rho < -0.99) p->rho = -0.99;

    // 5. Jumps
    if(p->lambda_j < 0.01) p->lambda_j = 0.01;
    if(p->lambda_j > 200.0) p->lambda_j = 200.0;
    
    if(p->sigma_j < 1e-4) p->sigma_j = 1e-4;
    if(p->sigma_j > 0.5) p->sigma_j = 0.5;
    
    // Feller Condition (Soft check, allow violation but penalize in optimizer via bounds)
    // We adjust sigma_v if it's wildly violating Feller to prevent explosion
    double feller = 2.0 * p->kappa * p->theta;
    if (p->sigma_v * p->sigma_v > feller * 2.0) {
        p->sigma_v = sqrt(feller * 2.0);
    }
}

void estimate_initial_params(double* returns, int n, SVCJParams* p) {
    // Calculate Realized Variance
    double sum = 0.0, sum_sq = 0.0;
    for(int i=0; i<n; i++) {
        sum += returns[i];
        sum_sq += returns[i] * returns[i];
    }
    double mean = sum / n;
    double var_daily = (sum_sq / n) - (mean * mean);
    double var_annual = var_daily * 252.0;

    // Set Defaults based on history
    p->mu = mean * 252.0;
    p->theta = var_annual;        // Start theta at realized variance
    p->kappa = 3.0;               // Moderate mean reversion
    p->sigma_v = sqrt(p->theta);  // Heuristic: vol of vol proportional to vol
    p->rho = -0.6;                // Typical equity leverage effect
    
    // Jump defaults
    p->lambda_j = 0.5;            // ~1 jump every 2 years base
    p->mu_j = -0.05;              // Downward jump bias
    p->sigma_j = 0.05;
    
    check_constraints(p);
}

// --- UKF Core ---

double ukf_log_likelihood(double* returns, int n, SVCJParams* p, double* out_spot_vol, double* out_jump_prob) {
    double ll = 0.0;
    double v = p->theta; // Initial state
    
    // Pre-calc jump variance contribution
    double var_j = p->mu_j*p->mu_j + p->sigma_j*p->sigma_j;
    
    for(int t=0; t<n; t++) {
        // --- 1. Prediction ---
        // Euler discretization of Heston variance process
        double v_pred = v + p->kappa * (p->theta - v) * DT;
        if(v_pred < 1e-7) v_pred = 1e-7; // Floor

        // --- 2. Observation Expectations ---
        // E[r] = (mu - 0.5*v - lambda*mu_j)*dt + lambda*mu_j*dt ... simplified:
        double drift = (p->mu - 0.5 * v_pred); 
        double y_hat = drift * DT; 
        double y = returns[t] - y_hat;

        // --- 3. Variance of Innovation ---
        // Total Variance = Diffusive + Jumps
        double S = v_pred * DT + p->lambda_j * var_j * DT;
        
        if (S < 1e-9) S = 1e-9;
        if (isnan(S) || isinf(S)) return -1e9;

        // --- 4. Update (Kalman Gain) ---
        // Simple linear estimator for QMLE
        double K = (p->rho * p->sigma_v * DT) / S;
        v = v_pred + K * y;
        
        // --- 5. Jump Detection (Instantaneous) ---
        // Mahalanobis distance
        double dist = (y*y) / S;
        double prob = 0.0;
        
        // Softmax-like probability for jump
        if(dist > 5.0) prob = 0.5 + 0.5 * tanh(dist - 5.0);
        else prob = p->lambda_j * DT;
        
        if(out_spot_vol) out_spot_vol[t] = sqrt(v_pred); // Report annualized vol
        if(out_jump_prob) out_jump_prob[t] = prob;

        // --- 6. Likelihood ---
        double step_ll = -0.5 * log(2 * M_PI * S) - 0.5 * (y*y)/S;
        ll += step_ll;
    }
    
    if(isnan(ll) || isinf(ll)) return -1e9;
    return ll;
}

// --- Optimization: Nelder-Mead Simplex ---
// A robust, derivative-free optimizer implemented purely in C

void optimize_svcj(double* returns, int n, SVCJParams* p, double* out_spot_vol, double* out_jump_prob) {
    // 1. Initialize center point
    estimate_initial_params(returns, n, p);
    
    // We optimize 5 core params: kappa, theta, sigma_v, rho, lambda_j
    // (mu, mu_j, sigma_j are often less sensitive or can be fixed to defaults for stability)
    
    int n_dim = 5;
    double simplex[6][5]; // n_dim + 1 points
    double scores[6];
    double param_arr[5];
    
    // Map struct to array
    param_arr[0] = p->kappa; param_arr[1] = p->theta; param_arr[2] = p->sigma_v;
    param_arr[3] = p->rho;   param_arr[4] = p->lambda_j;
    
    // Initialize Simplex (perturb each dim by 10%)
    for(int i=0; i<=n_dim; i++) {
        SVCJParams temp = *p;
        if(i > 0) {
            double val = param_arr[i-1];
            if(fabs(val) < 1e-4) val = 1e-4;
            // Perturb
            if(i==4) val += 0.1; // rho specific
            else val *= 1.15;
            
            // Map back to temp
            if(i==1) temp.kappa = val; else if(i==2) temp.theta = val;
            else if(i==3) temp.sigma_v = val; else if(i==4) temp.rho = val;
            else if(i==5) temp.lambda_j = val;
        }
        
        check_constraints(&temp);
        
        // Save into simplex
        simplex[i][0] = temp.kappa; simplex[i][1] = temp.theta; simplex[i][2] = temp.sigma_v;
        simplex[i][3] = temp.rho;   simplex[i][4] = temp.lambda_j;
        
        scores[i] = ukf_log_likelihood(returns, n, &temp, NULL, NULL);
    }
    
    // Nelder-Mead Loop
    for(int iter=0; iter<NM_ITER; iter++) {
        // 1. Sort simplex by score (descending because we maximize LL)
        // Bubble sort for small size 6
        for(int i=0; i<n_dim+1; i++) {
            for(int j=i+1; j<n_dim+1; j++) {
                if(scores[j] > scores[i]) { // Swap
                    double t_s = scores[i]; scores[i] = scores[j]; scores[j] = t_s;
                    for(int k=0; k<n_dim; k++) {
                        double t_p = simplex[i][k]; simplex[i][k] = simplex[j][k]; simplex[j][k] = t_p;
                    }
                }
            }
        }
        
        // Best is index 0, Worst is index n_dim
        
        // 2. Centroid of all but worst
        double centroid[5] = {0,0,0,0,0};
        for(int i=0; i<n_dim; i++) {
            for(int k=0; k<n_dim; k++) centroid[k] += simplex[i][k];
        }
        for(int k=0; k<n_dim; k++) centroid[k] /= n_dim;
        
        // 3. Reflection
        SVCJParams ref_p = *p;
        double reflected[5];
        for(int k=0; k<n_dim; k++) reflected[k] = centroid[k] + 1.0 * (centroid[k] - simplex[n_dim][k]);
        
        ref_p.kappa = reflected[0]; ref_p.theta = reflected[1]; ref_p.sigma_v = reflected[2];
        ref_p.rho = reflected[3]; ref_p.lambda_j = reflected[4];
        check_constraints(&ref_p);
        
        double ref_score = ukf_log_likelihood(returns, n, &ref_p, NULL, NULL);
        
        if(ref_score > scores[0]) {
            // Expansion
            SVCJParams exp_p = *p;
            double expanded[5];
            for(int k=0; k<n_dim; k++) expanded[k] = centroid[k] + 2.0 * (centroid[k] - simplex[n_dim][k]);
             
            exp_p.kappa = expanded[0]; exp_p.theta = expanded[1]; exp_p.sigma_v = expanded[2];
            exp_p.rho = expanded[3]; exp_p.lambda_j = expanded[4];
            check_constraints(&exp_p);
            
            double exp_score = ukf_log_likelihood(returns, n, &exp_p, NULL, NULL);
            
            if(exp_score > ref_score) {
                // Accept Expansion
                for(int k=0; k<n_dim; k++) simplex[n_dim][k] = expanded[k];
                scores[n_dim] = exp_score;
            } else {
                // Accept Reflection
                for(int k=0; k<n_dim; k++) simplex[n_dim][k] = reflected[k];
                scores[n_dim] = ref_score;
            }
        } else if(ref_score > scores[n_dim-1]) {
            // Accept Reflection
            for(int k=0; k<n_dim; k++) simplex[n_dim][k] = reflected[k];
            scores[n_dim] = ref_score;
        } else {
            // Contraction
            SVCJParams con_p = *p;
            double contracted[5];
            for(int k=0; k<n_dim; k++) contracted[k] = centroid[k] + 0.5 * (simplex[n_dim][k] - centroid[k]);
            
            con_p.kappa = contracted[0]; con_p.theta = contracted[1]; con_p.sigma_v = contracted[2];
            con_p.rho = contracted[3]; con_p.lambda_j = contracted[4];
            check_constraints(&con_p);
            
            double con_score = ukf_log_likelihood(returns, n, &con_p, NULL, NULL);
            
            if(con_score > scores[n_dim]) {
                for(int k=0; k<n_dim; k++) simplex[n_dim][k] = contracted[k];
                scores[n_dim] = con_score;
            } else {
                // Shrink
                 for(int i=1; i<=n_dim; i++) {
                     for(int k=0; k<n_dim; k++) {
                         simplex[i][k] = simplex[0][k] + 0.5 * (simplex[i][k] - simplex[0][k]);
                     }
                     // Re-eval
                     SVCJParams temp = *p;
                     temp.kappa = simplex[i][0]; temp.theta = simplex[i][1]; temp.sigma_v = simplex[i][2];
                     temp.rho = simplex[i][3]; temp.lambda_j = simplex[i][4];
                     check_constraints(&temp);
                     scores[i] = ukf_log_likelihood(returns, n, &temp, NULL, NULL);
                 }
            }
        }
    }
    
    // Result
    p->kappa = simplex[0][0]; p->theta = simplex[0][1]; p->sigma_v = simplex[0][2];
    p->rho = simplex[0][3]; p->lambda_j = simplex[0][4];
    
    // Final Run for Output
    ukf_log_likelihood(returns, n, p, out_spot_vol, out_jump_prob);
}

// --- Option Pricing ---
double normal_cdf(double x) { return 0.5 * erfc(-x * M_SQRT1_2); }

double bs_val(double S, double K, double T, double r, double sigma, int type) {
    if(T < 1e-5) return (type==1)? fmax(S-K,0):fmax(K-S,0);
    double d1 = (log(S/K) + (r + 0.5*sigma*sigma)*T) / (sigma*sqrt(T));
    double d2 = d1 - sigma*sqrt(T);
    if(type==1) return S*normal_cdf(d1) - K*exp(-r*T)*normal_cdf(d2);
    else return K*exp(-r*T)*normal_cdf(-d2) - S*normal_cdf(-d1);
}

void price_option_chain(double s0, double* strikes, double* expiries, int* types, int n_opts, SVCJParams* p, double spot_vol, double* out_prices) {
    // Merton Jump Diffusion sum
    double lambda = p->lambda_j;
    double m = p->mu_j; 
    double v_j = p->sigma_j;
    double lambda_p = lambda * (1.0 + m);
    double mu_drift = p->mu; // Risk free rate proxy

    for(int i=0; i<n_opts; i++) {
        double K = strikes[i]; double T = expiries[i]; int type = types[i];
        double price = 0.0;
        
        for(int k=0; k<15; k++) { // 15 terms
             double fact = 1.0; for(int j=1; j<=k; j++) fact *= j;
             double pois = exp(-lambda_p * T) * pow(lambda_p * T, k) / fact;
             
             if(pois < 1e-7 && k > 5) break; 
             
             double r_k = mu_drift - lambda*m + (k*log(1.0+m))/T;
             double sig_k = sqrt(spot_vol*spot_vol + (k*v_j*v_j)/T);
             
             price += pois * bs_val(s0, K, T, r_k, sig_k, type);
        }
        out_prices[i] = price;
    }
}