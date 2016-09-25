function [dat] = AcqFcnProb(dat,hyp)

% compute H here
dat.H         = kernelD(dat.rp,dat.xsamps,hyp,hyp.sigL);

nF            = size(dat.f0,2);
Ns            = size(dat.H,1);
Pf            = 0;
for j = 1:nF
    seps0     = hyp.seps(j);
    thet0     = hyp.thet(j);
    sInv      = diag(1./max(1e-6,diag(thet0*dat.s)+seps0^2));
    Minv      = dat.u*sInv*dat.v';
    Mf(:,j)   = Minv * (dat.f0(:,j) - mean(dat.f0(:,j))) + mean(dat.f0(:,j));
    fx        = dat.H * Mf(:,j);
    
    S         = thet0*(1 - sum(dat.H.*(dat.H * Minv), 2));
    S         = max(1e-6,S);

    Pf        = Pf - (fx-dat.fobj(j)).^2./(2*S.^2) - 1*log(S);
    clear fx; clear Minv; clear Mf;
end

dat.P = Pf; 