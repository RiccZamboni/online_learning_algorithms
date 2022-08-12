function pred = greedy_exp(y)

if isempty(y)
    pred = 0.5;
else
    pred = y(end);
end
