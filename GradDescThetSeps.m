function hyp = GradDescThetSeps(dat,hyp)

f0            = dat.f0;
nF            = size(f0,2);
u             = dat.u;
s             = dat.s;
v             = dat.v;
M             = dat.M;
for j = 1:nF
    seps0 = hyp.seps(j);
    thet0 = hyp.thet(j) + 0.25*(rand-0.5);
    logk = [];
    mm            = 0.1;
    gradThet      = 0;
    gradSeps      = 0;

    logp          = marglikelihood(seps0,thet0,M,u,s,v,f0(:,j));
    for k = 1:hyp.nopt
        sInv      = diag(1./max(1e-6,diag(s)+seps0^2));
        Minv      = u*sInv*v' / thet0;
        ginv      = sInv*v'*f0(:,j)*f0(:,j)'*u*sInv - thet0*sInv;
        gradK     = 1/thet0^2 * u*ginv*v';
        
        % move by gradient and see if marginal improves
        if hyp.thetscale==1
            gradThet  = 0.5*trace(gradK*M)*mm + gradThet*(1-mm);
            thetg = thet0 + gradThet*hyp.alpha;
        else
            thetg = thet0;
        end
        if hyp.sepscale==1
            gradSeps  = trace(seps0/thet0 * u*ginv*v')*mm + gradSeps*(1-mm);
            sepsg = max(0,seps0 + gradSeps*hyp.alpha);
        else      
            sepsg = seps0;
        end
        % compute marginal
        logpg     = marglikelihood(sepsg,thetg,M,u,s,v,f0(:,j));

        % stop gradients if not descending
        if logpg > logp
            logp  = logpg;
            seps0 = sepsg;
            thet0 = thetg;
        else
            break;
        end
    end

    if hyp.isGPU
        seps0 = gather(seps0);
        thet0 = gather(thet0);
    end
    hyp.seps(j)   = seps0;
    hyp.thet(j)   = thet0;
end

