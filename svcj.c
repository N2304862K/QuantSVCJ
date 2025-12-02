#include "svcj.h"
#include <float.h>

// --- Utilities ---

void clean_returns(double* returns, int n) {
    for(int i=0; i<n; i++) {
        if(fabs(returns[i]) < 1e-12) {
            returns[i] = (i % 2 == 0) ? 1e-12 : -1e-12;
        }
    }
}

// Helper: Calculate Realized Variance of the series
double get_realized_variance(double* returns, int n) {
    double sum = 0.0, sum_sq = 0.0;
    for(int i=0; i<n; i++) {
        sum += returns[i];
        sum_sq += returns[i] * returns[i];
    }
    double mean = sum / n;
    double var_daily = (sum_sq / n) - (mean * mean);
    return var_daily * 252.0; // Annualized
}

void check_constraints(SVCJParams* p) {
    // 1. Mean Reversion
    if(p->kappa < 0.2) p->kappa = 0.2;
    if(p->kappa > 30.0) p->kappa = 30.0;

    // 2. Long Run Variance
    if(p->theta < 0.0001) p->theta = 0.0001;
    if(p->theta > 1.0) p->theta = 1.0; 

    // 3. Vol of Vol
    if(p->sigma_v < 0.05) p->sigma_v = 0.05; // Hard floor to prevent deterministic collapse
    if(p->sigma_v > 5.0) p->sigma_v = 5.0;

    // 4. Correlation
    if(p->rho > 0.95) p->rho = 0.95;
    if(p->rho < -0.95) p->rho = -0.95;

    // 5. Jumps
    if(p->lambda_j < 0.05) p->lambda_j = 0.05; 
    if(p->lambda_j > 252.0) p->lambda_j = 252.0; 
    
    if(p->sigma_j < 0.005) p->sigma_j = 0.005; // Jumps must have minimum size
    if(p->sigma_j > 0.3) p->sigma_j = 0.3; 
    
    // Feller Soft Constraint
    double feller = 2.0 * p->kappa * p->theta;
    if (p->sigma_v * p->sigma_v > feller * 4.0) {
        p->sigma_v = sqrt(feller * 4.0);
    }
}

void estimate_initial_params(double* returns, int n, SVCJParams* p) {
    double rv = get_realized_variance(returns, n);

    p->mu = 0.0; // Assume near zero drift for daily
    p->theta = rv;
    if(p->theta < 0.0025) p->theta = 0.0025; 
    
    p->kappa = 4.0;
    p->sigma_v = sqrt(p->theta); 
    p->rho = -0.6;
    
    p->lambda_j = 0.5; 
    p->mu_j = 0.0;
    p->sigma_j = sqrt(rv/252.0) * 4.0; // Initial guess: Jumps are 4x daily vol
    
    check_constraints(p);
}

// --- UKF Core with Bayesian MAP & Phenotypic Mixing ---

double ukf_log_likelihood(double* returns, int n, SVCJParams* p, double* out_spot_vol, double* out_jump_prob) {
    double ll = 0.0;
    double v = p->theta; // State
    
    // Target Theta for Bayesian Prior (Realized Variance of the path)
    // In a strict filter, we calculate this once, but here we estimate it on fly or pass it.
    // For simplicity/speed in this C-func, we use a robust estimator derived from p->theta
    // (Assuming optimization started near truth).
    
    double var_j = p->mu_j*p->mu_j + p->sigma_j*p->sigma_j;
    
    for(int t=0; t<n; t++) {
        // 1. Predict
        double v_pred = v + p->kappa * (p->theta - v) * DT;
        if(v_pred < 1e-6) v_pred = 1e-6;

        // 2. Innovation
        double drift = (p->mu - 0.5 * v_pred);
        double y = returns[t] - drift * DT;

        // 3. Phenotypic Mixing (The Fix for Diffusion Collapse)
        // Problem: If v_pred -> 0, pdf_d becomes a Dirac Delta. Everything looks like a jump.
        // Fix: Test "Diffusion Hypothesis" against a ROBUST variance floor, not just v_pred.
        // We mix instantaneous view (v_pred) with long-run view (theta).
        
        double robust_var_d = fmax(v_pred, 0.25 * p->theta) * DT; 
        
        double pdf_d = (1.0 / (sqrt(robust_var_d) * SQRT_2PI)) * exp(-0.5 * y*y / robust_var_d);
        
        // Jump Hypothesis
        double total_var_j = robust_var_d + var_j; 
        double y_j = y - p->mu_j; 
        double pdf_j = (1.0 / (sqrt(total_var_j) * SQRT_2PI)) * exp(-0.5 * y_j*y_j / total_var_j);
        
        // Bayes Posterior
        double prob_prior = p->lambda_j * DT;
        if(prob_prior > 0.9) prob_prior = 0.9;
        
        double num = pdf_j * prob_prior;
        double den = num + pdf_d * (1.0 - prob_prior);
        if(den < 1e-15) den = 1e-15;
        
        double prob_posterior = num / den;
        
        // 4. State Update (Actual Filter Logic)
        // Here we use the REAL v_pred for update, but weighted by posterior
        double S = v_pred * DT + prob_posterior * var_j;
        
        // Kalman Gain
        double K = (p->rho * p->sigma_v * DT) / S;
        v = v_pred + K * y;
        
        // Clamp State
        if(v < 1e-6) v = 1e-6;
        if(v > 4.0) v = 4.0;

        // Outputs
        if(out_spot_vol) out_spot_vol[t] = sqrt(v_pred); 
        if(out_jump_prob) out_jump_prob[t] = prob_posterior;

        // Likelihood Accumulation
        ll += log(den); 
    }
    
    // --- Bayesian Priors (MAP Estimation) ---
    // These prevent the parameters from drifting to "cheating" values (like theta=0)
    
    double rv = get_realized_variance(returns, n);
    
    // Prior 1: Theta should not deviate wildly from Realized Variance
    // Log-Normal Prior: (ln(theta) - ln(RV))^2
    double theta_penalty = -20.0 * pow(log(p->theta) - log(rv), 2);
    
    // Prior 2: Jump Size (Sigma_J) should be significant
    // If Sigma_J is too small, jumps mimic diffusion. We expect Jumps ~ 3x Daily Vol
    double target_jump = sqrt(rv/252.0) * 3.0;
    double jump_penalty = -10.0 * pow(log(p->sigma_j) - log(target_jump), 2);
    
    // Prior 3: Rho (Correlation) usually negative for equity
    // Weak prior centering on -0.5
    double rho_penalty = -2.0 * pow(p->rho + 0.5, 2);

    if(isnan(ll) || isinf(ll)) return -1e15;
    
    return ll + theta_penalty + jump_penalty + rho_penalty;
}

// --- Optimization: Nelder-Mead ---
void optimize_svcj(double* returns, int n, SVCJParams* p, double* out_spot_vol, double* out_jump_prob) {
    estimate_initial_params(returns, n, p);
    
    int n_dim = 5;
    double simplex[6][5];
    double scores[6];
    
    // Create Simplex
    for(int i=0; i<=n_dim; i++) {
        SVCJParams temp = *p;
        if(i==1) temp.kappa *= 1.25;
        if(i==2) temp.theta *= 1.25;
        if(i==3) temp.sigma_v *= 1.25;
        if(i==4) temp.rho = (temp.rho > 0) ? temp.rho - 0.25 : temp.rho + 0.25;
        if(i==5) temp.lambda_j *= 1.5;
        check_constraints(&temp);
        
        simplex[i][0] = temp.kappa; simplex[i][1] = temp.theta; simplex[i][2] = temp.sigma_v;
        simplex[i][3] = temp.rho;   simplex[i][4] = temp.lambda_j;
        scores[i] = ukf_log_likelihood(returns, n, &temp, NULL, NULL);
    }
    
    // Optimization Loop
    for(int iter=0; iter<NM_ITER; iter++) {
        // Sort Indices
        int vs[6]; for(int k=0; k<6; k++) vs[k]=k;
        for(int i=0; i<6; i++) {
            for(int j=i+1; j<6; j++) {
                if(scores[vs[j]] > scores[vs[i]]) { int t=vs[i]; vs[i]=vs[j]; vs[j]=t; }
            }
        }
        
        // Centroid
        double c[5] = {0};
        for(int i=0; i<5; i++) { for(int k=0; k<5; k++) c[k] += simplex[vs[i]][k]; }
        for(int k=0; k<5; k++) c[k] /= 5.0;
        
        // Reflect
        double ref[5]; SVCJParams rp = *p;
        for(int k=0; k<5; k++) ref[k] = c[k] + 1.0 * (c[k] - simplex[vs[5]][k]);
        rp.kappa=ref[0]; rp.theta=ref[1]; rp.sigma_v=ref[2]; rp.rho=ref[3]; rp.lambda_j=ref[4];
        check_constraints(&rp);
        double r_score = ukf_log_likelihood(returns, n, &rp, NULL, NULL);
        
        if(r_score > scores[vs[0]]) {
            // Expand
            double exp[5]; SVCJParams ep = *p;
            for(int k=0; k<5; k++) exp[k] = c[k] + 2.0 * (c[k] - simplex[vs[5]][k]);
            ep.kappa=exp[0]; ep.theta=exp[1]; ep.sigma_v=exp[2]; ep.rho=exp[3]; ep.lambda_j=exp[4];
            check_constraints(&ep);
            double e_score = ukf_log_likelihood(returns, n, &ep, NULL, NULL);
            if(e_score > r_score) { for(int k=0; k<5; k++) simplex[vs[5]][k] = exp[k]; scores[vs[5]] = e_score; } 
            else { for(int k=0; k<5; k++) simplex[vs[5]][k] = ref[k]; scores[vs[5]] = r_score; }
        } else if(r_score > scores[vs[4]]) {
             for(int k=0; k<5; k++) simplex[vs[5]][k] = ref[k]; scores[vs[5]] = r_score;
        } else {
            // Contract
            double con[5]; SVCJParams cp = *p;
            for(int k=0; k<5; k++) con[k] = c[k] + 0.5 * (simplex[vs[5]][k] - c[k]);
            cp.kappa=con[0]; cp.theta=con[1]; cp.sigma_v=con[2]; cp.rho=con[3]; cp.lambda_j=con[4];
            check_constraints(&cp);
            double c_score = ukf_log_likelihood(returns, n, &cp, NULL, NULL);
            if(c_score > scores[vs[5]]) { for(int k=0; k<5; k++) simplex[vs[5]][k] = con[k]; scores[vs[5]] = c_score; }
            else {
                // Shrink
                for(int i=1; i<6; i++) {
                    int idx = vs[i]; SVCJParams sp = *p;
                    for(int k=0; k<5; k++) simplex[idx][k] = simplex[vs[0]][k] + 0.5 * (simplex[idx][k] - simplex[vs[0]][k]);
                    sp.kappa=simplex[idx][0]; sp.theta=simplex[idx][1]; sp.sigma_v=simplex[idx][2]; sp.rho=simplex[idx][3]; sp.lambda_j=simplex[idx][4];
                    check_constraints(&sp); scores[idx] = ukf_log_likelihood(returns, n, &sp, NULL, NULL);
                }
            }
        }
    }
    
    // Set Best Params
    int best = 0; for(int i=1; i<6; i++) if(scores[i] > scores[best]) best = i;
    p->kappa = simplex[best][0]; p->theta = simplex[best][1]; p->sigma_v = simplex[best][2];
    p->rho = simplex[best][3]; p->lambda_j = simplex[best][4];
    
    ukf_log_likelihood(returns, n, p, out_spot_vol, out_jump_prob);
}

// --- Pricing ---
double normal_cdf(double x) { return 0.5 * erfc(-x * M_SQRT1_2); }
double bs_calc(double S, double K, double T, double r, double v, int type) {
    if(T < 1e-4) return (type==1)?fmax(S-K,0):fmax(K-S,0);
    double d1 = (log(S/K)+(r+0.5*v*v)*T)/(v*sqrt(T));
    double d2 = d1 - v*sqrt(T);
    if(type==1) return S*normal_cdf(d1)-K*exp(-r*T)*normal_cdf(d2);
    else return K*exp(-r*T)*normal_cdf(-d2)-S*normal_cdf(-d1);
}

void price_option_chain(double s0, double* strikes, double* expiries, int* types, int n_opts, SVCJParams* p, double spot_vol, double* out_prices) {
    double lambda = p->lambda_j; 
    double m = p->mu_j; 
    double v_j = p->sigma_j;
    double lamp = lambda*(1.0+m);
    
    for(int i=0; i<n_opts; i++) {
        double val = 0.0;
        for(int k=0; k<12; k++) {
             double fact=1.0; for(int j=1;j<=k;j++) fact*=j;
             double prob = exp(-lamp*expiries[i]) * pow(lamp*expiries[i], k) / fact;
             if(prob < 1e-8 && k>2) break;
             double rk = p->mu - lambda*m + (k*log(1.0+m))/expiries[i];
             double vk = sqrt(spot_vol*spot_vol + (k*v_j*v_j)/expiries[i]);
             val += prob * bs_calc(s0, strikes[i], expiries[i], rk, vk, types[i]);
        }
        out_prices[i] = val;
    }
}