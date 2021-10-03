function P = randpoints_nsphere(Npoints, Ndim, radius) 

X = randn(Npoints, Ndim);
X = X ./ sqrt(sum(X.^2, 2)); % project to the surface of the Ndim-sphere
Npoints = size(X, 1);

% Scale with random radius
R = radius*rand(Npoints,Ndim);
P = R.*X;
