function p=plot_regret(regret)

figure();
time_horizon = size(regret, 2);

p=plot(1:time_horizon, regret);

ylabel('Regret');
xlabel('t');