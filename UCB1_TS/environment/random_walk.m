function [r, x] = random_walk(R, a)

persistent state

n_arms = length(R);
if isempty(state)
    state = 0.5 * ones(n_arms, 1);
end
state = max(min(state + 1 / 3 * randn(n_arms, 1), 1), 0);

r = state(a);
x = [];
