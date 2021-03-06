function  [xprop,pmax,dat,hyp]       = OptProb5d(dat,hyp)

%%%% simulate observed points 
%    (in this case pull from grid values f)
%dat.f0        = fD(dat.f,dat.xsamps)';
%  pull from GP with hyperparameters hypPrior
xprop             = dat.xsamps(:,end);
x             = dat.xsamps(:,1:end-1);
if isempty(dat.H)
    % xsamps-observed kernel
    M             = kernelD(x,x,hyp,hyp.sigL);
    dat.M         = M;
    %%%% invert samps-to-samps kernel
    [u s v]       = svd(M);
    dat.u         = u;
    dat.s         = s;
end
H             = kernelD(xprop,x,dat.hypPrior,dat.hypPrior.sigL);
nF            = size(dat.f0,2);
for j = 1:nF
    seps0     = dat.hypPrior.seps(j);
    thet0     = dat.hypPrior.thet(j);
    sInv      = diag(1./max(1e-6,diag(thet0*dat.s)+seps0^2));
    Minv      = dat.u*sInv*dat.u';
    fmean     = mean(dat.f0(1:size(x,2),j));
    Mf(:,j)   = Minv * (dat.f0(1:size(x,2),j) - fmean) + fmean;
    fx        = H * Mf(:,j);
    S         = thet0*(1 - sum(H.*(H * Minv), 2));
    S         = max(1e-6,S);
    keyboard;
    dat.f0(size(dat.xsamps,2),j) = fx + fmean + randn*S;
end


%%%% optimize hyperparameters and compute acquisition fcn
[dat,hyp]      = ProbHypOpt(dat,hyp); 
if hyp.isGPU
    dat.P = gather(dat.P);
end

%%%% acq func max(P) for next pt to propose
[pmax,imax]    = max(dat.P(:));
xprop          = dat.rp(:,imax);
