data {
  int<lower=0> N;
  int<lower=0> N_analyte;
  int<lower=0> N_cond;
  int<lower=0> N_comp;
  vector[N] y;
  array[N] int<lower=0> condition;
  array[N] int<lower=0> analyte;
  array[N_comp, 2] int<lower=0> comparisons;
  array[N_cond, N_analyte] real means;
  array[N_cond, N_analyte] real stds;
}
parameters {
  array[N_cond, N_analyte] real est_mean;
  array[N_cond, N_analyte] real<lower=0> est_std;
  real<lower=0> normality;
  real<lower=0> hyper_std;
  real<lower=0> hyper_fc_mean;
  real<lower=0> hyper_fc_std;
}
transformed parameters{
  array[N_comp, N_analyte] real fold_changes;
  array[N_cond, N_analyte] real newmu;
  for (a in 1:N_analyte) {
    for (c in 1:N_cond) {
      newmu[c,a] = est_mean[c,a] * stds[c,a];
      newmu[c,a] += means[c,a];
    }
  }
  for (comp in 1:N_comp) {
    for (a in 1:N_analyte) {
      fold_changes[comp,a] = newmu[comparisons[comp,1],a] / newmu[comparisons[comp,2],a];
    }
  }
}
model {
  hyper_std ~ lognormal(0.1,1);
  hyper_fc_mean ~ lognormal(1,0.5);
  hyper_fc_std ~ lognormal(0.5,0.5);
  normality ~ lognormal(1,1);
  for (a in 1:N_analyte) {
    for (c in 1:N_cond) {
        est_mean[c,a] ~ normal(0,0.1);
        est_std[c,a] ~ exponential(hyper_std);
    }
  }
  for (i in 1:N) {
    y[i] ~ student_t(normality, est_mean[condition[i], analyte[i]], est_std[condition[i], analyte[i]]);
  }
  for (comp in 1:N_comp) {
    for (a in 1:N_analyte) {
      fold_changes[comp,a] ~ lognormal(hyper_fc_mean,hyper_fc_std);
    }
  }
}


