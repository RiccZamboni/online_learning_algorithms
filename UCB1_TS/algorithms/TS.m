function ind = TS(cum_r, N)
    alpha = 1+cum_r;
    beta = 1+N-cum_r;
    theta = betarnd(alpha, beta);
    [~, ind] = max(theta); 
end
