function logp      = marglikelihood(seps0,thet0,M,u,s,f0)

f0            = bsxfun(@minus,f0,mean(f0,1));
for j = 1:size(f0,2)
    sInv      = diag(1./max(1e-6,thet0(j)*diag(s)+seps0(j)^2));
    Minv      = u*sInv*u';
    My        = (thet0(j)*M+seps0(j)^2*eye(size(M)));
    logp(j)   = -.5*f0(:,j)'*Minv*f0(:,j)-.5*sum(log(thet0(j)*diag(s)+seps0(j)^2));
end
  
        