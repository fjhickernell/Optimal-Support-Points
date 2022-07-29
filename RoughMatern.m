function [sqdisc,grad] = RoughMatern(x,flag,beta)
n = size(x,1);
if flag(1)
    kernel = exp(-beta*abs(x-x'));
    eta = (2  - exp(-beta*x) - exp(beta*(x-1)))/beta;
    xi = (2/beta^2)*(beta - 1 + exp(-beta));
    sqdisc = xi - (2/n)*sum(eta) + (1/n^2) * sum(sum(kernel));
else
    sqdisc = NaN;
end
if flag(2)
    etaprime = exp(-beta * x) - exp(beta*(x-1));
    zetamat = -beta*exp(-beta * abs(x-x')) .* sign(x-x');
    grad = -2*etaprime + (2/n)*sum(zetamat,2) - (1/n) * diag(zetamat);
else
    grad = NaN(n,1);
end

