% RADIAL BASIS FUNCTION (RBF) con ampiezza SIGMA
function K = rbf_kernel(U,V,sigma)
    gamma = 1 ./ (2*(sigma^2));
    K = exp(-gamma .* pdist2(U,V,'euclidean').^2);
end