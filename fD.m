function fp = fD(f,inds)

inds = inds';
D  = ndims(f)-1;
idx = inds(:,1);
SX  = 1;
for i = 2:D
    SX = SX*size(f,i);
    idx = idx + (inds(:,i)-1)*SX;
end
%keyboard;

fp = f(:,idx);

end