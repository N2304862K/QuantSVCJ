#ifndef SVCJ_H
#define SVCJ_H

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    double kappa;    // Mean reversion
    double theta;    // Long-run variance
    double sigma_v;  // Vol of Vol
    double rho;      // Correlation
    double lambda;   // Jump Intensity
    double mu_j;     // Mean Jump Size
    double sigma_j;  // Jump Size Std Dev
} SVCJParams;

typedef struct {
    SVCJParams p;
    double spot_vol;    // Robust Spot Volatility
    double jump_prob;   // Probability current return is a jump
    double error;       // Final optimization error
} SVCJResult;

// Mode: 0 = History Only (UKF), 1 = Joint (UKF + Options)
SVCJResult optimize_svcj(double* returns, int n_ret, double dt,
                         double* strikes, double* prices, double* T_exp, int n_opts,
                         double S0, double r, int mode);

#ifdef __cplusplus
}
#endif

#endif