function ind = UCB1(rew, pulls, t)

N = sum(pulls);
cum_r = sum(rew);

hat_R = cum_r ./ N;
B = sqrt(2 * log(t) ./ N);
if t <= length(cum_r)
    ind = t;
else
    U = min(1, hat_R + B);
    
    [~, ind] = max(U);
end
