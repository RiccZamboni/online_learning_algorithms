function ind = SW_TS(rew, pulls, t, tau)
    if t>tau
        N = sum(pulls(t-tau:t,:));
        cum_r = sum(rew(t-tau:t,:));
    else
        N = sum(pulls);
        cum_r = sum(rew);
    end
    theta = betarnd(cum_r+1, N-cum_r+1);
    [~, ind] = max(theta);
end