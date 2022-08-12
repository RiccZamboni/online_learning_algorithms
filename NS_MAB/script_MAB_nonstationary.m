clear;
clear CDTUCB.m
clc;
close all;
addpath(genpath('.'));

%% Mab stochastic environment
rng(0)

algo = 'cdtucb'; % ucb1, ts, sw\_ts, cdtucb

R = [0.2 0.3 0.7 0.5; ...
    0.1 0.2 0.4 0.6; ...
    0.9 0.5 0.7 0.1; ...
    0.3 0.4 0.5 0.8];


T = 3000;
tau_sw_ts = 200;
breakpoints = [1000 1500 2000];

plot_evolution(R, breakpoints, T);

n_arms = length(R);

% UCB1
pulls = zeros(T, n_arms);
reward = zeros(T, n_arms);
pulls_ucb = zeros(T, n_arms);
reward_ucb = zeros(T, n_arms);

fig = figure();
ind = zeros(T,1);
change = zeros(T,1);
rewards = zeros(T,1);
rewards_ucb = zeros(T,1);

for tt = 1:T
    % Algortithm choice
    if strcmp(algo,'ucb1')
        ind(tt) = UCB1(reward, pulls, tt);
    elseif strcmp(algo,'ts')
        ind(tt) = TS(reward, pulls, tt);
    elseif strcmp(algo,'sw\_ts')
        ind(tt) = SW_TS(reward, pulls, tt, tau_sw_ts);
    else
        [ind(tt),change(tt)] = CDTUCB(reward, pulls, tt,T);
    end
    %Plot bounds
    ind_ucb(tt) = UCB1(reward_ucb, pulls_ucb, tt);
    
    %Reward
    %rewards(tt) = stochastic_env(R, ind(tt));
    rewards(tt) = stochastic_nonst_env(R, breakpoints, tt, ind(tt));
    rewards_ucb(tt) = stochastic_nonst_env(R, breakpoints, tt, ind_ucb(tt));
    
    %Update statistics
    pulls(tt, ind(tt)) = pulls(tt, ind(tt)) + 1;
    pulls_ucb(tt, ind_ucb(tt)) = 1;
    reward(tt, ind(tt)) = reward(tt, ind(tt)) + rewards(tt);
    reward_ucb(tt, ind_ucb(tt)) = reward_ucb(tt, ind_ucb(tt)) + rewards_ucb(tt);
end

%% Plot Pseudo Regret

breakpoints = [0 breakpoints T];
%breakpoints = [0 T];

n_phase = length(breakpoints);
hold on
for ii = 1:(n_phase-1)
    tt_ind = (breakpoints(ii)+1):breakpoints(ii+1);
    exp_rew = R(ii, :);
    pseudo_regret(tt_ind) = max(exp_rew) - exp_rew(ind(tt_ind));
    pseudo_regret_ucb(tt_ind) = max(exp_rew) - exp_rew(ind_ucb(tt_ind));
end

plot(cumsum(pseudo_regret),'b');
plot(cumsum(pseudo_regret_ucb),'g');
legend(algo,'ucb1 baseline')
for ii = 2:length(breakpoints)-1
    xline(breakpoints(ii),'r');
end
hold off
