function ind = UCB1(cum_r, N, t)
    if t<=length(N)
        ind=t;
    else
        mu = mean(cum_r./N,1);
        b = sqrt(2*log(t)./N);
        bounds = mu + b;
        [~, ind]= max(bounds);
    end