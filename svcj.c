#include "svcj.h"
#include <float.h>

// --- Utilities ---

void clean_returns(double* returns, int n) {
    for(int i=0; i<n; i++) {
        // Prevent absolute zero (singularity risk)
        if(fabs(returns[i]) < 1e-12) {
            returns[i] = (i % 2 == 0) ? 1e-12 : -1e-12;
        }
    }
}

void check_constraints(SVCJParams* p) {
    // 1. Mean Reversion
    if(p->kappa < 0.1) p->kappa = 0.1;
    if(p->kappa > 25.0) p->kappa = 25.0;

    // 2. Long Run Variance (Annualized)
    // 0.0001 (1% vol) to 0.64 (80% vol)
    if(p->theta < 0.0001) p->theta = 0.0001;
    if(p->theta > 0.64) p->theta = 0.64; 

    // 3. Vol of Vol
    if(p->sigma_v < 0.01) p->sigma_v = 0.01;
    if(p->sigma_v > 5.0) p->sigma_v = 5.0;

    // 4. Correlation
    if(p->rho > 0.90) p->rho = 0.90;
    if(p->rho < -0.90) p->rho = -0.90;

    // 5. Jumps
    if(p->lambda_j < 0.05) p->lambda_j = 0.05; // At least one jump every 20 years
    if(p->lambda_j > 252.0) p->lambda_j = 252.0; // Max 1 jump per day avg
    
    if(p->sigma_j < 0.001) p->sigma_j = 0.001;
    if(p->sigma_j > 0.2) p->sigma_j = 0.2; // Cap jump std dev size
    
    // Feller Soft Constraint
    double feller = 2.0 * p->kappa * p->theta;
    if (p->sigma_v * p->sigma_v > feller * 3.0) {
        p->sigma_v = sqrt(feller * 3.0);
    }
}

void estimate_initial_params(double* returns, int n, SVCJParams* p) {
    double sum = 0.0, sum_sq = 0.0;
    for(int i=0; i<n; i++) {
        sum += returns[i];
        sum_sq += returns[i] * returns[i];
    }
    double mean = sum / n;
    double var_daily = (sum_sq / n) - (mean * mean);
    double var_annual = var_daily * 252.0;

    // Initialize sensible defaults
    p->mu = mean * 252.0;
    p->theta = var_annual;
    if(p->theta < 0.0025) p->theta = 0.0025; // Floor at 5% vol
    
    p->kappa = 4.0;
    p->sigma_v = sqrt(p->theta); 
    p->rho = -0.5;
    
    p->lambda_j = 0.5; // Start conservative
    p->mu_j = -0.02;
    p->sigma_j = sqrt(var_daily) * 3.0; // Jump size scaled to daily vol
    
    check_constraints(p);
}

// --- UKF Core with Bayesian Jump Filter ---

double ukf_log_likelihood(double* returns, int n, SVCJParams* p, double* out_spot_vol, double* out_jump_prob) {
    double ll = 0.0;
    double v = p->theta; // State: Annualized Variance
    
    // Constants for Jump Distribution
    double var_j = p->mu_j*p->mu_j + p->sigma_j*p->sigma_j;
    
    for(int t=0; t<n; t++) {
        // 1. Predict (Euler)
        double v_pred = v + p->kappa * (p->theta - v) * DT;
        if(v_pred < 1e-6) v_pred = 1e-6;

        // 2. Innovation
        double drift = (p->mu - 0.5 * v_pred);
        double y_hat = drift * DT;
        double y = returns[t] - y_hat;

        // 3. Diffusive Variance component
        double var_d = v_pred * DT;
        
        // 4. Bayesian Jump Probability
        // Hypothesis 0: Diffusion Only (Normal(0, var_d))
        // Hypothesis 1: Jump + Diffusion (Normal(mu_j, var_d + var_j))
        // We approximate Jump density as Gaussian for the filter weight
        
        double pdf_d = (1.0 / (sqrt(var_d) * SQRT_2PI)) * exp(-0.5 * y*y / var_d);
        
        double total_var_j = var_d + var_j; // Variance if jump occurs
        double y_j = y - p->mu_j; // Centered on jump mean
        double pdf_j = (1.0 / (sqrt(total_var_j) * SQRT_2PI)) * exp(-0.5 * y_j*y_j / total_var_j);
        
        // Prior probability of jump in this step
        double prob_prior = p->lambda_j * DT;
        if(prob_prior > 1.0) prob_prior = 1.0;
        
        // Posterior Probability (Bayes Rule)
        double numerator = pdf_j * prob_prior;
        double denominator = numerator + pdf_d * (1.0 - prob_prior);
        if(denominator < 1e-12) denominator = 1e-12;
        
        double prob_posterior = numerator / denominator;
        
        // 5. Update State (QMLE approximation)
        // Weighted variance of innovation
        double S = var_d + prob_posterior * var_j; 
        
        double K = (p->rho * p->sigma_v * DT) / S;
        v = v_pred + K * y;
        
        // Clamp
        if(v < 1e-6) v = 1e-6;
        if(v > 2.0) v = 2.0;

        // Outputs
        if(out_spot_vol) out_spot_vol[t] = sqrt(v_pred); // Annualized Vol
        if(out_jump_prob) out_jump_prob[t] = prob_posterior;

        // Likelihood accumulation
        ll += log(denominator); 
    }
    
    if(isnan(ll) || isinf(ll)) return -1e15;
    return ll;
}

// --- Optimization: Nelder-Mead ---

void optimize_svcj(double* returns, int n, SVCJParams* p, double* out_spot_vol, double* out_jump_prob) {
    estimate_initial_params(returns, n, p);
    
    int n_dim = 5;
    double simplex[6][5];
    double scores[6];
    
    // Initial Simplex Construction
    for(int i=0; i<=n_dim; i++) {
        SVCJParams temp = *p;
        
        if(i==1) temp.kappa *= 1.2;
        if(i==2) temp.theta *= 1.2;
        if(i==3) temp.sigma_v *= 1.2;
        if(i==4) temp.rho = (temp.rho > 0) ? temp.rho - 0.2 : temp.rho + 0.2;
        if(i==5) temp.lambda_j *= 1.5;
        
        check_constraints(&temp);
        
        // Map back to array for logic
        simplex[i][0] = temp.kappa; simplex[i][1] = temp.theta; simplex[i][2] = temp.sigma_v;
        simplex[i][3] = temp.rho;   simplex[i][4] = temp.lambda_j;
        
        scores[i] = ukf_log_likelihood(returns, n, &temp, NULL, NULL);
    }
    
    // Iterations
    for(int iter=0; iter<NM_ITER; iter++) {
        // Sort
        int vs[6]; for(int k=0; k<6; k++) vs[k]=k;
        // Simple bubble sort indices based on scores (Descending)
        for(int i=0; i<6; i++) {
            for(int j=i+1; j<6; j++) {
                if(scores[vs[j]] > scores[vs[i]]) { int t=vs[i]; vs[i]=vs[j]; vs[j]=t; }
            }
        }
        
        // Centroid of top 5
        double c[5] = {0};
        for(int i=0; i<5; i++) {
            for(int k=0; k<5; k++) c[k] += simplex[vs[i]][k];
        }
        for(int k=0; k<5; k++) c[k] /= 5.0;
        
        // Reflect
        double ref[5];
        SVCJParams rp = *p;
        for(int k=0; k<5; k++) ref[k] = c[k] + 1.0 * (c[k] - simplex[vs[5]][k]);
        
        rp.kappa=ref[0]; rp.theta=ref[1]; rp.sigma_v=ref[2]; rp.rho=ref[3]; rp.lambda_j=ref[4];
        check_constraints(&rp);
        double r_score = ukf_log_likelihood(returns, n, &rp, NULL, NULL);
        
        if(r_score > scores[vs[0]]) {
            // Expand
            double exp[5];
            SVCJParams ep = *p;
            for(int k=0; k<5; k++) exp[k] = c[k] + 2.0 * (c[k] - simplex[vs[5]][k]);
            ep.kappa=exp[0]; ep.theta=exp[1]; ep.sigma_v=exp[2]; ep.rho=exp[3]; ep.lambda_j=exp[4];
            check_constraints(&ep);
            double e_score = ukf_log_likelihood(returns, n, &ep, NULL, NULL);
            
            if(e_score > r_score) {
                for(int k=0; k<5; k++) simplex[vs[5]][k] = exp[k];
                scores[vs[5]] = e_score;
            } else {
                for(int k=0; k<5; k++) simplex[vs[5]][k] = ref[k];
                scores[vs[5]] = r_score;
            }
        } else if(r_score > scores[vs[4]]) {
             for(int k=0; k<5; k++) simplex[vs[5]][k] = ref[k];
             scores[vs[5]] = r_score;
        } else {
            // Contract
            double con[5];
            SVCJParams cp = *p;
            for(int k=0; k<5; k++) con[k] = c[k] + 0.5 * (simplex[vs[5]][k] - c[k]);
            cp.kappa=con[0]; cp.theta=con[1]; cp.sigma_v=con[2]; cp.rho=con[3]; cp.lambda_j=con[4];
            check_constraints(&cp);
            double c_score = ukf_log_likelihood(returns, n, &cp, NULL, NULL);
            
            if(c_score > scores[vs[5]]) {
                for(int k=0; k<5; k++) simplex[vs[5]][k] = con[k];
                scores[vs[5]] = c_score;
            } else {
                // Shrink
                for(int i=1; i<6; i++) {
                    int idx = vs[i];
                    SVCJParams sp = *p;
                    for(int k=0; k<5; k++) {
                        simplex[idx][k] = simplex[vs[0]][k] + 0.5 * (simplex[idx][k] - simplex[vs[0]][k]);
                    }
                    sp.kappa=simplex[idx][0]; sp.theta=simplex[idx][1]; sp.sigma_v=simplex[idx][2];
                    sp.rho=simplex[idx][3]; sp.lambda_j=simplex[idx][4];
                    check_constraints(&sp);
                    scores[idx] = ukf_log_likelihood(returns, n, &sp, NULL, NULL);
                }
            }
        }
    }
    
    // Best Params
    int best = 0;
    for(int i=1; i<6; i++) if(scores[i] > scores[best]) best = i;
    
    p->kappa = simplex[best][0];
    p->theta = simplex[best][1];
    p->sigma_v = simplex[best][2];
    p->rho = simplex[best][3];
    p->lambda_j = simplex[best][4];
    
    // Final Run
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