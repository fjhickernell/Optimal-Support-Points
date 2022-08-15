%% Check gradient

gail.InitializeWorkspaceDisplay

beta = 1; %shape parameter
%kernelfun = @(x,flag) RoughMatern(x,flag,beta);
kernelfun = @(x,flag) SqExpon(x,flag,beta);

n = 2;
xdes0 = ((1:n)' - 1/2)/n;
xdes1 = xdes0;
[sqdisc0,grad] = kernelfun(xdes0,[1 1]);
h = 0.0001;
appxgrad = grad;

for jj = 1:n
    xdes1 = xdes0;
    xdes1(jj) = xdes1(jj) + h;
    sqdisc1 = kernelfun(xdes1,[1 0]);
    appxgrad(jj) = n*(sqdisc1 - sqdisc0)/h;
end

[grad appxgrad]

n = 200;
xdes0 = ((1:n)' - 1/2)/n;
[sqdisc0] = kernelfun(xdes0,[1 0])
