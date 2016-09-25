
function K = kernelD(xp0,yp0,hyp,sigL)
ktype = hyp.ktype;

D  = size(xp0,1);
N  = size(xp0,2); 
M  = size(yp0,2);

% split M into chunks if on GPU
if hyp.isGPU
    K=gpuArray.zeros(N,M,'single');
    cs  = 50;
else
    K= zeros(N,M,'single');
    cs  = M;
end

for i = 1:ceil(M/cs)
    ii = [((i-1)*cs+1):min(M,i*cs)];
    mM = length(ii);
    xp = repmat(xp0,1,1,mM);
    yp = reshape(repmat(yp0(:,ii),N,1),D,N,mM);

    if strcmp(ktype,'matern')
        r  = squeeze(sum((xp - yp).^2,1));
        Kn = (1 + sqrt(3)*r/sigL) .* exp(-sqrt(3)*r/sigL);
    else
        Kn = exp(-(xp - yp).^2 * (1./(2*sigL.^2)));
    end
    K(:,ii)  = squeeze(Kn); 
end


 