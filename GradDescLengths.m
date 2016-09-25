function [dat,hyp] = GradDescLengths(dat,hyp)

f0            = dat.f0;
nF            = size(f0,2);
u             = dat.u;
s             = dat.s;
v             = dat.v;
M             = dat.M;
x             = dat.xsamps;
D             = size(x,1);
% initial marginal likelihood of observed pts
% and the xdiff mats for the gradients
logp          = 0;
for j = 1:nF
    seps0     = hyp.seps(j);
    thet0     = hyp.thet(j);
    logpj     = marglikelihood(seps0,thet0,M,u,s,v,f0(:,j));
    logp      = logp + logpj;    
end
for j = 1:D
    xj        = x(j,:);
    xj        = repmat(xj,numel(xj),1)-repmat(xj',1,numel(xj));
    xdiff{j}  = xj.^2;
end

gradLens       = zeros(D,1,'single');
if hyp.isGPU
    gradLens   = gpuArray.zeros(D,1,'single');
end
len0              = max(0.5,hyp.sigL + 1*(rand(size(hyp.sigL))-0.5));    
for k = 1:hyp.nopt
    mm            = 0.1;

    % compute gradient - sum over feature dims
    gradLen       = zeros(D,1,'single');
    if hyp.isGPU
        gradLen   = gpuArray.zeros(D,1,'single');
    end
    for j = 1:nF
        seps0     = hyp.seps(j);
        thet0     = hyp.thet(j);
        sInv      = diag(1./max(1e-6,diag(s)+seps0^2));
        Minv      = u*sInv*v' / thet0;
        ginv      = sInv*v'*f0(:,j)*f0(:,j)'*u*sInv - thet0*sInv;
        gradK     = 1/thet0^2 * u*ginv*v';
        for d = 1:D
            gradLen(d) = gradLen(d) + 0.5*trace(thet0*gradK*(M.*xdiff{d})/len0(d)^3);
        end
    end        
    gradLens      = gradLens*(1-mm) + mm*gradLen;
    leng          = max(0.5,len0 + gradLens*hyp.alpha);
    logpg         = 0;
    Mg            = kernelD(x,x,hyp,leng);
    [ug sg vg]    = svd(Mg);
    for j = 1:nF
        seps0     = hyp.seps(j);
        thet0     = hyp.thet(j);
        logpj     = marglikelihood(seps0,thet0,Mg,ug,sg,vg,f0(:,j));
        logpg     = logpg + logpj;
    end
    %logk(k)       = gather(logpg);
    %disp([logp logpg gradLens']);
   
    % stop gradients if not descending
    if logpg > logp
        logp      = logpg;
        len0      = leng;
        M         = Mg;
        u         = ug;
        s         = sg;
        v         = vg;
    else
        break;
    end
end
        
if hyp.isGPU
    len0  = gather(len0);
end
hyp.sigL   = len0;
dat.M      = M;
dat.u      = u;
dat.s      = s;
dat.v      = v;
