function ind = TS(rew, pulls, t)

N = sum(pulls);
cum_r = sum(rew);

theta = betarnd(cum_r+1,N-cum_r+1);
[~, ind] = max(theta);
