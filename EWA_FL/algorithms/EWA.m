function pred = EWA(w, exp_advice)
    pred = sum(w.*exp_advice)/sum(w);