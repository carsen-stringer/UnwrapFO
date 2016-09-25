function logp      = marglikelihood(seps0,thet0,M,u,s,v,f0)

sInv      = diag(1./max(1e-6,diag(s)+seps0^2));
Minv      = u*sInv*v' / thet0;
My        = (M+seps0^2*eye(size(M)))*thet0;
logp      = -.5*f0'*Minv*f0-.5*log(det(My)+1e-6);
  
        