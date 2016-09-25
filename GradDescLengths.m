function [dat,hyp] = GradDescLengths(dat,hyp)

f0            = dat.f0;
nF            = size(f0,2);
u             = dat.u;
s             = dat.s;
M             = dat.M;
x             = dat.xsamps;
D             = size(x,1);
% initial marginal likelihood of observed pts
% and the xdiff mats for the gradients
logpj         = marglikelihood(hyp.seps,hyp.thet,M,u,s,f0);
logp          = sum(logpj);
for j = 1:D
    xj        = x(j,:);
    xj        = repmat(xj,numel(xj),1)-repmat(xj',1,numel(xj));
    xdiff{j}  = xj.^2;
end

gradLens       = zeros(D,1,'single');
if hyp.isGPU
    gradLens   = gpuArray.zeros(D,1,'single');
end
%hyp.sigL       = max(0.5,hyp.sigL + 0.5*(rand(size(hyp.sigL))-0.5));    
for k = 1:hyp.nopt
    mm            = 0.1;

    % compute gradient - sum over feature dims
    [~,~,gradLen] = gradlogp(dat,hyp,xdiff);
    gradLens      = gradLens*(1-mm) + mm*gradLen;
    leng          = max(0.5,hyp.sigL + gradLens*hyp.alpha);
    Mg            = kernelD(x,x,hyp,leng);
    [ug sg vg]    = svd(Mg);
    logpj         = marglikelihood(hyp.seps,hyp.thet,Mg,ug,sg,f0);
    logpg         = sum(logpj);

    logk(k)       = gather(logpg);
    %disp([logp logpg gradLens']);
   
    % stop gradients if not descending
    if 0
        if logpg > logp
            logp      = logpg;
            hyp.sigL  = leng;
            dat.M     = Mg;
            dat.u     = ug;
            dat.s     = sg;
        else
            break;
        end
    end
end

if hyp.isGPU
    hyp.sigL  = gather(hyp.sigL);
end