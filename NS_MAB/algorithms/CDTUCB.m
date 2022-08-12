function [ind, change] = CDTUCB(reward, pulls, tt, T)

n_arms = size(reward, 2);
m = 40;
h = 10/8*log(T);
alpha = 0.01*n_arms*sqrt(1/T*log(T)); %0.01;
epsilon = 0.5;
ksi = 1; % for UCB
change = false;

persistent gplus
persistent gminus
persistent last_change
persistent bar_mu
persistent idx_last_pulls 

if isempty(gplus)
    gplus = zeros(1, n_arms);
    gminus = zeros(1, n_arms);
    last_change = ones(1, n_arms);
    bar_mu = zeros(1, n_arms);
    idx_last_pulls = ones(1, n_arms);
end

% Reset CDT if necessary
for ii=1:n_arms
    if gplus(ii) > h || gminus(ii) > h
        %'yes'
        %tt
        last_change(ii) = tt;
        gplus(ii) = 0;
        gminus(ii) = 0;
        idx_last_pulls(ii) = tt;
        change = true;
    end
end

%trigger = gplus > h | gminus > h;
%last_change(trigger) = tt;
%gplus(trigger) = 0;
%gminus(trigger) = 0;

% Compute the pulls for each arm
avail_pulls = zeros(1, n_arms);
for ii = 1:n_arms
    avail_pulls(ii) = sum(pulls(last_change(ii):tt, ii));
end

% Check if an arm has less than m pulls, if so pull it
if any(avail_pulls<m)
    ind = find(avail_pulls<m,1);
else
    % Compute bar_mu for all the arms having exactly m pulls
    for ii=1:n_arms
        bar_mu(ii) = mean(reward(idx_last_pulls(ii):tt-1,ii));
    end
    

    % Update the CDT using the last reward generated
    gplus = max(0, gplus + reward(size(reward,1),:)-bar_mu-epsilon);
    gminus = max(0, gminus - reward(size(reward,1),:)+bar_mu-epsilon);
        
    

    % Pull arms according to UCB with probability 1-\alpha
    if rand<alpha
        ind = randi(n_arms);
        
    else
        N = sum(pulls(last_change:tt,:));
        hat_R = sum(reward((last_change:tt),:)) ./ N;
        B = sqrt(ksi * log(sum(N)) ./ N);
        U = min(1, hat_R + B);
        [~, ind] = max(U);
    end
    
    % update index of last m pulls for arm ind
    idx_last_pulls(ind) = find(pulls(idx_last_pulls(ind)+1:tt,ind)>0,1);
end
end
