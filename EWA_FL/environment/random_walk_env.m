function [y, l, exp_l] = random_walk_env(hat_y, f, loss_f)

persistent state

if isempty(state)
    state = rand();
end
state = max(min(state + 1 / 1 * randn(), 1), 0);

y = state;
l = loss_f(hat_y, state);
exp_l = loss_f(f, state);
