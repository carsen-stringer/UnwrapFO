function [dat,hyp]      = ProbHypOpt(dat,hyp)

%%%% simulations that have been observed
x             = dat.xsamps;

% xsamps-observed kernel
M             = kernelD(x,x,hyp,hyp.sigL);
dat.M         = M;

%%%% invert samps-to-samps kernel
[u s v]       = svd(M);
dat.u         = u;
dat.s         = s;
dat.v         = v;
clear u v s M x;

%%%% also optimize hyperparameters!
[dat,hyp]     = MaxMargProb(dat,hyp);

%%%% compute acquisition function!
[dat]         = AcqFcnProb(dat,dat.hypPrior);
