clear;
clc;
close all;
addpath(genpath('.'));

%% Mab stochastic environment
alg = 'UCB1'; %UCB1, TS

R = [0.2 0.3 0.7 0.5];
n_arms = length(R);

%Environment

T = 3000;

% UCB1

N = zeros(1, n_arms);
cum_r = zeros(1, n_arms);

ind = zeros(T,1);
rewards = zeros(T,1);
fig=figure();
%%
for tt = 1:T
    % Algortithm choice
    if strcmp(alg,'UCB1')
        ind(tt) = UCB1(cum_r, N, tt);
        %Plot bounds
        fig = plot_UCB1bound(fig, T, tt, cum_r, N, R, ind(tt));
    end
    if strcmp(alg,'TS')
        ind(tt) = TS(cum_r, N);
    end
    
    %Reward
    rewards(tt) = stochastic_env(R, ind(tt));
    
    %Update statistics
    N(ind(tt)) = N(ind(tt)) + 1;
    cum_r(ind(tt)) = cum_r(ind(tt)) + rewards(tt);
end

%% Plot Pseudo Regret
pseudo_regret = cumsum(max(R)-R(ind));
plot_regret(pseudo_regret);

Delta_vec = max(R) - R;
Delta_vec = Delta_vec(Delta_vec > 0);
UpperBound = 8*log(1:T)*sum(1./Delta_vec)+(1+pi^2/3)*sum(Delta_vec);

hold on
plot(1:T, UpperBound, 'g');
title("Pseudo regret over one run")
legend({'Pseudo regret' 'UCB1 Upper bound'}, 'Location', 'NorthWest');

%% Compute expected pseudo regret
n_rep = 10;
ind = zeros(T,n_rep);
rewards = zeros(T, n_rep);
for rr = 1:n_rep    
    N = zeros(1, n_arms);
    cum_r = zeros(1, n_arms);

    for tt = 1:T
        % Algorithm choice
        if strcmp(alg,'UCB1')
            ind(tt, rr) = UCB1(cum_r, N, tt);
        end
        if strcmp(alg,'TS')
            ind(tt, rr) = TS(cum_r, N);
        end
        %Reward
        rewards(tt, rr) = stochastic_env(R, ind(tt, rr));
        
        %Update statistics
        N(ind(tt, rr)) = N(ind(tt, rr)) + 1;
        cum_r(ind(tt, rr)) = cum_r(ind(tt, rr)) + rewards(tt, rr);
    end
end


%% Plot Expected Pseudo Regret
pseudo_regret =cumsum(max(R)-R(ind));
p=plot_regret(pseudo_regret');
h(1)=p(1);
hold on
p2=plot(1:T, UpperBound, 'g');
h(2) = p2(1);
title("Pseudo regret over " + n_rep + " runs")
legend(h, {'Pseudo regret' 'UCB1 Upper bound'}, 'Location', 'NorthWest');
