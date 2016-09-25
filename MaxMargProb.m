function [dat,hyp]          = MaxMargProb(dat,hyp)

nF            = size(dat.f0,2);
Ns            = size(dat.H,1);

% hyperparameter optimization through gradients of logp
if (hyp.thetscale+hyp.sepscale+hyp.lenscale)>0 && hyp.nopt>0
    if size(dat.f0,1)>hyp.burnin
        %%%% joint optimization of lengths and amps and noise
        if hyp.lenscale==1 
            [dat,hyp] = GradDescAll(dat,hyp);
        %%%% optimization of amps and observation noise
        else
            hyp       = GradDescThetSeps(dat,hyp);
        end
    end
end