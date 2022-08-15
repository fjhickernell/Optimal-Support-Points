function [sqdisc,grad] = SqExpon(x,flag,beta)
n = size(x,1);
if flag(1)
    kernel = exp(-(beta*(x-x')).^2);
    eta = (sqrt(pi)/(2*beta))*(erf(beta*x) + erf(beta*(1-x)));
    xi = (1/beta^2)*(- 1 + exp(-beta^2) + beta*sqrt(pi)*erf(beta));
    sqdisc = xi - (2/n)*sum(eta) + (1/n^2) * sum(sum(kernel));
else
    sqdisc = NaN;
end
if flag(2)
    etaprime = exp(-(beta * x).^2) - exp(-(beta * (1-x)).^2);
    zetamat = (-2*beta^2)*exp(-(beta*(x-x')).^2) .* (x-x');
    omegavec = zeros(size(x,1),1);
    grad = -2*etaprime + (2/n)*sum(zetamat,2) - (2/n) * diag(zetamat) + ...
        (1/n)*omegavec;
else
    grad = NaN(n,1);
end