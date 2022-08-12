function pred = window_exp(y)

if isempty(y)
    pred = 0.5;
elseif length(y) <= 20
    pred = mean(y);
else
    pred = mean(y(end-19:end));
end
