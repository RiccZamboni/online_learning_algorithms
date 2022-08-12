function [y, l, exp_l] = random_env(hat_y, f, loss_f)

y = rand();
l = loss_f(hat_y, y);
exp_l = loss_f(f, y);