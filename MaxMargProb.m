function [dat,hyp]          = MaxMargProb(dat,hyp)

nF            = size(dat.f0,2);
Ns            = size(dat.H,1);

%%%% optimize kernel amplitude and observation noise
if size(dat.f0,1)>25 && hyp.nopt>0 && (hyp.thetscale+hyp.sepscale)>0
    hyp       = GradDescThetSeps(dat,hyp);
end

%%%% optimize kernel lengths
if size(dat.f0,1)>25 && hyp.nopt>0 && hyp.lenscale==1
    [dat,hyp] = GradDescLengths(dat,hyp);
end
% reoptimize scale parameters
if size(dat.f0,1)>25 && hyp.nopt>0 && (hyp.thetscale+hyp.sepscale)>0
    hyp       = GradDescThetSeps(dat,hyp);
end


dat.H         = kernelD(dat.rp,dat.xsamps,hyp,hyp.sigL);

