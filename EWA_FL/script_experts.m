clear;
clc;
close all;
addpath(genpath('.'));

%% Expert Setting

alg = 'EWA'; %EWA, FL

%Environment
T = 20000;


pred = zeros(T,1);
loss = zeros(T,1);
y_t = [];

experts = {@constant_exp, @greedy_exp, @window_exp};
n_exp = length(experts);
exp_loss = zeros(n_exp, T);

% EWA
hat_w = ones(n_exp, 1);
%eta= 0.5;
eta = sqrt(8*log(n_exp)/(T));
%%

for tt = 1:T
    % Experts Advice
    for ii = 1:3
        exp_advice(ii, 1) = experts{ii}(y_t);
    end
    
    % Algorithm choice   
    if strcmp(alg, 'EWA')
        pred(tt) = EWA(hat_w, exp_advice);
    elseif strcmp(alg, 'FL')
        pred(tt) = FL(exp_loss, exp_advice);
    end

    %Reward
    [y, loss(tt), exp_loss(:, tt)] = random_env(pred(tt), exp_advice, @quad_loss);
    
    %Update statistics
    if strcmp(alg, 'EWA')
        update = exp(-eta*exp_loss(:, tt));
        hat_w = hat_w.*update;
        hat_w = hat_w./sum(hat_w);
    end
    y_t = [y_t y];
end


%% Plot Losses
alg_loss = cumsum(loss);

exp_losses = cumsum(exp_loss');

figure();
plot([alg_loss exp_losses]);
legend('Agent loss', 'constant\_exp', 'greedy\_exp', 'window\_exp')

%% Plot Regret
regret = cumsum(loss) - min(cumsum(exp_loss'), [], 2);
plot_regret(regret');
hold on

% Plot bounds
%% Write the expression of the bound
if strcmp(alg, 'EWA')
    %plot(arrayfun(@(x) log(n_exp)/eta + eta*x/8, 1:T));
    plot(arrayfun(@(x) sqrt(x*log(n_exp)/2), 1:T));
elseif strcmp(alg, 'FL')
    plot(1:T, 8*(log(1:T)+1))
end

legend('regret', 'bound')
