function [r, x] = stochastic_nonst_env(R, breakpoints, tt, a)

phase = 1 + sum(tt > breakpoints);

n_arms = size(R, 2);
all_r = rand(1, n_arms) < R(phase, :);

r = all_r(a);
x = [];
