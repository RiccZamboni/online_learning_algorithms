function plot_regret(regret)

figure();
time_horizon = size(regret, 2);

plot(1:time_horizon, regret);

ylabel('Regret');
xlabel('t');