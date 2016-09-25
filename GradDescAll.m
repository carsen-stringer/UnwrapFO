function [dat,hyp] = GradDescAll(dat,hyp)

f0            = dat.f0;
x             = dat.xsamps;
nF            = size(f0,2);
D             = size(x,1);
u             = dat.u;
s             = dat.s;
M             = dat.M;

mm            = 0.1;
gradThet      = zeros(nF,1);
gradSeps      = zeros(nF,1);

% initial marginal likelihood of observed pts
% and the xdiff mats for the gradients
logp          = marglikelihood(hyp.seps,hyp.thet,M,u,s,f0);
for j = 1:D
    xj        = x(j,:);
    xj        = repmat(xj,numel(xj),1)-repmat(xj',1,numel(xj));
    xdiff{j}  = xj.^2;
end

gradLens       = zeros(D,1,'single');
if hyp.isGPU
    gradLens   = gpuArray.zeros(D,1,'single');
    hyp.seps   = gpuArray(hyp.seps);
    hyp.thet   = gpuArray(hyp.thet);
    hyp.sigL   = gpuArray(hyp.sigL);
end

for k = 1:hyp.nopt
    [gradS,gradT,gradLen] = gradlogp(dat,hyp,xdiff);

    %%%% move by gradient and see if marginal improves
    % theta gradient
    if hyp.thetscale==1
        gradThet  = gradT*mm + gradThet*(1-mm);
        thetg     = hyp.thet + gradThet*hyp.alpha;
    else
        thetg     = hyp.thet;
    end
    % seps gradient
    if hyp.sepscale==1
        gradSeps  = gradS*mm + gradSeps*(1-mm);
        sepsg     = max(0,hyp.seps + gradSeps*hyp.alpha);
    else      
        sepsg     = hyps.seps;
    end
    % length gradient
    gradLens      = gradLen* mm + gradLens*(1-mm);
    leng          = max(0.5,hyp.sigL + gradLens*hyp.alpha);
    Mg            = kernelD(x,x,hyp,leng);
    [ug sg vg]    = svd(Mg);

    % compute marginal
    logpg         = marglikelihood(sepsg,thetg,Mg,ug,sg,f0);

    hyp.seps  = sepsg;
    hyp.thet  = thetg;
    hyp.sigL  = leng;
    
    % stop gradients if not descending
    if 0
    ginc = 0;
    if sum(logpg) > sum(logp)
        ginc      = ginc+1;
        logp      = logpg;
        hyp.seps  = sepsg;
        hyp.thet  = thetg;
        hyp.sigL  = leng;
    else  
        for j = 1:nF
            if logpg(j) > logp(j)
                logp(j)  = logpg(j);
                hyp.seps(j) = sepsg(j);
                hyp.thet(j) = thetg(j);
                ginc  = ginc+1;
            end
        end
    end
    if ginc == 0
        break;
    end
    end
end

if hyp.isGPU
    hyp.seps = gather(hyp.seps);
    hyp.thet = gather(hyp.thet);
    hyp.sigL = gather(hyp.sigL);
end
    
