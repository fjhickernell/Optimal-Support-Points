function [sqdisc,grad] = RoughMatern(x,flag)
n = size(x,1);
if flag(1)
    kernel = exp(-abs(x-x'));
    eta = 2  - exp(-x) - exp(x-1);
    xi = 2*exp(-1);
    sqdisc = xi - (2/n)*sum(eta) + (1/n^2) * sum(sum(kernel));
else
    sqdisc = NaN;
end
if flag(2)
    etaprime = exp(-x) - exp(x-1);
    zetamat = -exp(-abs(x-x')) .* sign(x-x');
    grad = -2*etaprime + (2/n)*sum(zetamat,2) - (1/n) * diag(zetamat);
else
    grad = NaN(n,1);
end

