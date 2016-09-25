
addpath('~/grive/important_functions/');

if ~exist('xB')
    load('~/GPUopti2/LOESS/fiveD/fGridNoNan.mat');
end
D = size(xB,1);


%%%% reshape objectives and f's
f             = [fmu(1:3,:,:,:,:,:);fac(1:3,:,:,:,:,:);fnc];
fobj          = [Fmu(:,1:3)';Fac(:,1:3)';Fnc(:)'];
nF            = size(f,1);  % number of features for BO
% z-score f by fobj
f             = (f(:,:) - repmat(mean(fobj,2),1,size(f,2)^D))./ ...
                repmat(std(fobj,1,2),1,size(f,2)^D);
fobj          = (fobj - repmat(mean(fobj,2),1,size(fobj,2)))./ ...
                repmat(std(fobj,1,2),1,size(fobj,2));
f             = reshape(f,[size(f,1) ngrid ngrid ngrid ngrid ngrid]);
fobj0         = fobj;

%%%% create grid of parameter space
[x1,x2,x3,x4,x5] = ndgrid([1:ngrid]);
xD            = zeros(5,ngrid,ngrid,ngrid,ngrid,ngrid,'single');
xD(1,:)       = x1(:);
xD(2,:)       = x2(:);
xD(3,:)       = x3(:);
xD(4,:)       = x4(:);
xD(5,:)       = x5(:);
xD            = xD(:,:);
if isGPU
    xD        = gpuArray(xD);
end

%%%% initial points for BO (isamps)
ninit  = 1;
if ~exist('isamps')
    isamps    = 8*ones(D,1,'single');
end

%%%% bayesian optimization settings
nIter         = 80;       % number of iterations
nopt          = 5;        % number of hyperparameter opts
alpha0        = 5e-3;     % learning rate for hyperopt

%%%% hyperparameter settings (initial)
ktype         = 'gaussian';
seps0         = .2 * ones(nF,1);
sigD0         = 2 * ones(D,1); % different length for each input dim
thet0         = 0.5*ones(nF,1);
clear hyp;
hyp.sigL      = sigD0; % length scale parameter
hyp.seps      = seps0; % observation noise
hyp.thet      = thet0; % kernel scale
hyp.nopt      = nopt; % number of hyperparameter steps
hyp.UB        = ngrid; % upper bound of x-grid
hyp.alpha     = alpha0;
hyp.ktype     = ktype; % kernel type
hyp.isGPU     = isGPU;
hyp.lenscale  = 1;    % optimize length scale too?
hyp.thetscale = 1;    % optimize theta
hyp.sepscale  = 1;    % optimize seps
hyp.burnin    = 25;   % how many iterations before hyperparameter
                      % optimization starts


xF=[]; CF=[]; CFk=[];
for iDat = 5%2:44
    tic;
    ixp = isamps;
    % objective function depends on iDat
    fobj = fobj0(:,iDat);
    % cost function smoothed and min found
    ccf = squeeze(sum(bsxfun(@minus,f,fobj).^2,1));
    ccsm = my_conv2(ccf,.5,[1:ndims(ccf)]);
    %ccsm = ccf;
    [csmin,ixmin] = min(ccsm(:));
    fprintf('%d goal is cost = %2.4f <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<\n',iDat,(csmin));
    ixx   = sub2ind(size(ccsm),isamps(1,:),isamps(2,:),isamps(3,:),isamps(4,:),isamps(5,:));

    % initialize costs
    cstep           = [];
    cstep(1:ninit)  = ccsm(ixx);
    cmin            = min(cstep);

    if isGPU
        ixp  = gpuArray(single(ixp));
        fobj = gpuArray(single(fobj));
    end
  
    xk            = zeros(D,nIter+ninit);
    xk(:,1:ninit) = isamps;

    hyp.sigL      = sigD0; % length scale parameter
    hyp.seps      = seps0; % observation noise
    hyp.thet      = thet0; % kernel scale

    clear dat;
    dat.hypPrior  = hyp;

    for k = ninit+1:nIter
        % initialize dat
        if k==ninit+1
            dat.H    = [];
            dat.P    = [];
            dat.fobj = fobj;
            dat.f    = f;
            dat.xsamps = ixp;
            dat.rp   = xD;
        end
        [xprop,pprop,dat,hyp]   = OptProb5d(dat,hyp);
        if isGPU
            xprop   = gather(xprop);
            pprop   = gather(pprop);
        end
        ixp         = [ixp xprop]; % samples done
        dat.xsamps  = ixp;
        xk(:,k)     = xprop;

        % compute minimum
        cstep(k)  = ccsm(xprop(1),xprop(2),xprop(3),xprop(4),xprop(5));
        cmin(k)   = min(cstep);
        fprintf('%d\t%2.4f\t%2.2f\t%d',k,cmin(k),pprop,sum(ccsm(:)<cmin(k)))
        fprintf('\t%d',xprop);
        fprintf('\n');
        fprintf('\t %2.4f',[mean(hyp.seps(:)) ...
                            mean(hyp.thet(:))]);
        fprintf('\n');
        fprintf('\t %1.1f',(hyp.sigL));
        fprintf('\n');

            
    end
    toc;
    fprintf('%d\t%2.4f\t%2.4f\n',k,cmin(k),csmin)
    
    cmin(length(cmin)+1) = csmin;
    cmin(length(cmin)+1) = sum(ccsm(:)<cmin(k+1));
    cmin(length(cmin)+1) = iDat;

    CF  = [CF;cmin];
    CFk = [CFk;cstep];
    xF  = [xF;xk];
    
    save('CFht','CF','xF','CFk');
end
