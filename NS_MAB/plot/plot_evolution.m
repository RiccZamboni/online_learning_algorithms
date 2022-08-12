function plot_evolution(R, breakpoints, T)

n_arms = size(R, 2);
breakpoints = [1 breakpoints T];


x = 1:T;

y = zeros(n_arms, T);

for ii = 1:(length(breakpoints)-1)
    phase_len = breakpoints(ii+1) - breakpoints(ii);
    y(:, breakpoints(ii):(breakpoints(ii+1)-1)) = repmat(R(ii, :)', 1, phase_len);
end

figure();
plot(x,y);
ylabel('\mu_{i,t}');
xlabel('t');