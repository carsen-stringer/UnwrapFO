function [gradSeps,gradThet,gradLen] = gradlogp(dat,hyp,xdiff)

u             = dat.u;
s             = dat.s;
D             = size(dat.xsamps,1);
nF            = size(dat.f0,2);
% compute gradient - sum over feature dims
gradLen       = zeros(D,1,'single');
gradSeps      = zeros(nF,1,'single');
gradThet      = zeros(nF,1,'single');
if hyp.isGPU
    gradSeps  = gpuArray.zeros(nF,1,'single');
    gradThet  = gpuArray.zeros(nF,1,'single');
    gradLen   = gpuArray.zeros(D,1,'single');
end
len0          = hyp.sigL;
for j = 1:nF
    seps0     = hyp.seps(j);
    thet0     = hyp.thet(j);
    sInv      = diag(1./max(1e-6,thet0*diag(s)+seps0^2));
    Kinv      = u*sInv*u';
    Kmat      = u*s*sInv*u';
    fmat      = Kinv*dat.f0(:,j)*dat.f0(:,j)'-eye(size(Kinv));
    if nargin>2
        for d = 1:D
            dKdL = thet0*(dat.M.*xdiff{d})/len0(d)^3;
            gradLen(d) = gradLen(d) + 0.5*trace(fmat*Kinv*dKdL);
        end
    else
        gradLen = [];
    end
    
    gradThet(j) = 0.5*trace(fmat*Kmat);
    gradSeps(j) = seps0*trace(fmat*Kinv);

end        
