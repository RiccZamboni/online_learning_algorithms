function fig = plot_UCB1bound(fig, T, t, cum_r, N, R, selected_arm)

colors = ['m' 'g' 'b' 'k'];
hat_R = cum_r ./ N;
B = sqrt(2 * log(t) ./ N);
U = min(1, hat_R + B);
n_arms = length(cum_r);

if t < 10 || mod(t, T / 20) == 0
    clf;
    hold on;
    for ii = 1:n_arms
        errorbar(ii, hat_R(ii), U(ii) - hat_R(ii), '.', 'Color', colors(mod(ii-1, 4)+1));
        text(ii, 1.2, num2str(N(ii)), 'Color', colors(mod(ii-1, 4)+1));
    end
    hold on;
    plot(1:n_arms, R, 'rx'.', 'LineWidth', 3);
    ylim([0,1.3]);
    
    ax = gca;
    ax.XTick = 1:n_arms;
    title(['Arm selected a_' num2str(selected_arm) ', t = ' num2str(t)]);
    drawnow();
    pause();
end