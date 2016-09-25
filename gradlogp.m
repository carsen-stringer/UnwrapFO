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
    sInv      = diag(1./max(1e-6,diag(thet0*s)+seps0^2));
    Minv      = u*sInv*u';
    Kinv      = u*s*sInv*u';
    gradK     = Minv*dat.f0(:,j)*dat.f0(:,j)'-eye(size(Minv));
    if nargin>2
        for d = 1:D
            gradLen(d) = gradLen(d) + 0.5*trace(thet0*gradK*Minv*(dat.M.*xdiff{d})/len0(d)^3);
        end
    else
        gradLen = [];
    end
    
    gradThet(j) = 0.5*trace(gradK*Kinv);
    gradSeps(j) = seps0*trace(gradK*Minv);

end        
