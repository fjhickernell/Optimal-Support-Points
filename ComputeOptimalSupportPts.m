%% Compute Optimal Support Points

gail.InitializeWorkspaceDisplay

beta = 1; %shape parameter
%kernelfun = @(x,flag) RoughMatern(x,flag,beta);
kernelfun = @(x,flag) SqExpon(x,flag,beta);
ntry = 2000; %number of gradient descent steps


% mmax = 3;
% nvec = 2.^(0:mmax);
nvec = 1:8;
maxn=max(nvec);
nveclength = length(nvec);
finaldisc(nveclength,1) = 0;
initdisc(nveclength,1) = 0;
deschange(nveclength,1) = 0;
discimprov(nveclength,1) = 0;
extdes = zeros(maxn,1);
stopcrit = 1e-14;
nov2 = 0;
xdesall(maxn,nveclength) = 0;


for jj = 1:nveclength
    n = nvec(jj)
    xdestry = zeros(n,3);
    xdesinit = (1:n)'/n - 1/(2*n);
%     if jj == 1
%         xdesinit = (1:n)'/n - 1/(2*n);
%     else
%         extdes = [0; xdesall(1:nold,jj-1); 1];
%         gaps = diff(extdes);
%         nnew = n - nold;
%         [~,wh] = maxk(gaps,nnew);
%         xdesinit = [xdes; (extdes(wh)+extdes(wh+1))/2];
%     end

    xdes = xdesinit;
    [sqdiscval,grad] = kernelfun(xdes,[1,1]); %initial squared discrepancy and gradient
    initdisc(jj) = sqrt(sqdiscval);
    stepsize = 1/n;
    [xdes,sqdiscval,grad,stepsize] = ...
        optimizeDesign(kernelfun,xdes,sqdiscval,grad,stepsize,ntry,stopcrit);
    finaldisc(jj) = sqrt(sqdiscval); %best discrepancy found
    deschange(jj) = norm(xdes-xdesinit)/norm(xdesinit); %change in initial design
    discimprov(jj) = finaldisc(jj)/initdisc(jj); %change in initial design
    xdesall(1:n,jj) = xdes;
    nold = n;   
end
figure
axis([0 1 0 nveclength+1])
hold on
for jj = 1:nveclength
    n = nvec(jj);
    plot(xdesall(1:n,jj),jj*ones(n,1),'.')
end

figure
loglog(nvec,finaldisc/finaldisc(1),'.', ...
    nvec,nvec(1)./nvec,'--', ...
    nvec,nvec(1)./nvec.^2,'--')
hold on
loglog(nvec,initdisc/finaldisc(1),'o','markersize',10)
xlabel('\(n\)')
ylabel('Normalized Discrepancy')

figure
loglog(nvec,discimprov,'.')
xlabel('\(n\)')
ylabel('Final Discrepancy Relative to Initial')

figure
loglog(nvec,deschange,'.')
xlabel('\(n\)')
ylabel('Relative \(\ell_2\) Design Change')




