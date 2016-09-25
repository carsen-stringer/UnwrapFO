function  [xprop,pmax,dat,hyp]       = OptProb5d(dat,hyp)

%%%% simulate observed points 
%    (in this case pull from grid values f)
dat.f0        = fD(dat.f,dat.xsamps)';

%%%% optimize hyperparameters and compute acquisition fcn
[dat,hyp]      = ProbHypOpt(dat,hyp); 
if hyp.isGPU
    dat.P = gather(dat.P);
end

%%%% acq func max(P) for next pt to propose
[pmax,imax]    = max(dat.P(:));
xprop          = dat.rp(:,imax);
