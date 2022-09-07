clear;
clc;
close all;
addpath(genpath('.'));

%% Expert Setting

alg = 'EWAt'; %EWA, FL, EWA_t

%Environment
T = 20000;


pred = zeros(T,1);
loss = zeros(T,1);
y_t = [];

experts = {@constant_exp, @greedy_exp, @window_exp};
n_exp = length(experts);
exp_loss = zeros(n_exp, T);

% EWA
w_hat = ones(n_exp, 1);
eta = sqrt(8*log(n_exp)/(T));
eta_t = arrayfun(@(x) sqrt(8*log(n_exp)/(x)), 1:T);
%%

for tt = 1:T
    % Experts Advice
    for ii = 1:3
        exp_pred(ii, 1) = experts{ii}(y_t);
    end
    
    % Algorithm choice   
    if strcmp(alg, 'EWA')
        pred(tt) = EWA(w_hat, exp_pred);
    elseif strcmp(alg, 'EWAt')
        pred(tt) = EWA(w_hat, exp_pred);
    elseif strcmp(alg, 'FL')
        pred(tt) = FL(exp_loss, exp_pred);
    end

    %Reward
    [y, loss(tt), exp_loss(:, tt)] = random_env(pred(tt), exp_pred, @quad_loss);
    
    %Update statistics
    if strcmp(alg, 'EWA')
        local_update = exp(-eta*exp_loss(:, tt));
        w_hat = w_hat.*local_update;
        w_hat = w_hat./sum(w_hat);
    elseif strcmp(alg, 'EWAt')
        local_update = exp(-eta_t(tt)*exp_loss(:, tt));
        w_hat = w_hat.*local_update;
        w_hat = w_hat./sum(w_hat);
    end
    y_t = [y_t y];
end


%% Plot Losses
alg_loss = cumsum(loss);

exp_losses = cumsum(exp_loss');

figure();
plot([alg_loss exp_losses]);
title("Losses")
legend('Agent loss', 'constant\_expert loss', 'greedy\_expert loss', 'window\_expert loss')

%% Plot Regret
regret = cumsum(loss) - min(cumsum(exp_loss'), [], 2);
plot_regret(regret');
hold on

% Plot bounds
%% Write the expression of the bound
if strcmp(alg, 'EWA')
    %plot(arrayfun(@(x) log(n_exp)/eta + eta*x/8, 1:T));
    plot(arrayfun(@(x) sqrt(x*log(n_exp)/2), 1:T));
elseif strcmp(alg, 'EWAt')
    plot(arrayfun(@(x) sqrt(x*log(n_exp)/2) + sqrt(log(n_exp)/8), 1:T))
elseif strcmp(alg, 'FL')
    plot(1:T, 8*(log(1:T)+1))
end
title("Regrets")
legend(['regret of' ' ' alg] , ['bound of' ' ' alg] )
