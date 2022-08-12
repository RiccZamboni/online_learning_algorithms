function [y, l, exp_l] = stochastic_env(hat_y, f, loss_f)

mu = 0.6;
y = rand() < mu;

l = loss_f(hat_y, y);
exp_l = loss_f(f, y);
