function pred = FL(exp_loss, exp_advice)
    idx = find(sum(exp_loss,2)==min(sum(exp_loss,2)));
    pred = mean(exp_advice(idx));