function [r, x] = stochastic_env(R, a)

n_arms = length(R);
all_r = rand(1, n_arms) < R;

r = all_r(a);
x = [];