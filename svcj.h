#ifndef SVCJ_H
#define SVCJ_H
#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    double kappa; double theta; double sigma_v; double rho;
    double lambda; double mu_j; double sigma_j;
} SVCJParams;

typedef struct {
    SVCJParams p;
    double spot_vol; double jump_prob; double error;
} SVCJResult;

SVCJResult optimize_svcj(double* returns, int n_ret, double dt,
                         double* strikes, double* prices, double* T_exp, int n_opts,
                         double S0, double r, int mode);
#ifdef __cplusplus
}
#endif
#endif