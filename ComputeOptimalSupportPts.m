%% Compute Optimal Support Points

gail.InitializeWorkspaceDisplay

kernelfun = @RoughMatern;
ntry = 50; %number of gradient descent steps


mmax = 8;
nvec = 2.^(0:mmax);
maxn=2^mmax;
nveclength = length(nvec);
finaldisc(nveclength,1) = 0;
initdisc(nveclength,1) = 0;
deschange(nveclength,1) = 0;
discimprov(nveclength,1) = 0;
extdes = zeros(maxn,1);
stopcrit =  1e-10;
nov2 = 0;


for jj = 1:nveclength
    n = nvec(jj);
    xdestry = zeros(n,3);
    xdes = (1:n)'/n - 1/(2*n);
%     newpts = (xdes(1:nov2) + [0; xdes(1:nov2-1)])/2;
%     xdes = [xdes(1:nov2,1) newpts];
    xdesinit = xdes;
    [sqdiscval,grad] = kernelfun(xdes,[1,1]); %initial squared discrepancy and gradient
    initdisc(jj) = sqrt(sqdiscval);
    stepsize = 1/n;
    %disp('Beginning with a design of')
    %disp(xdes)
    %disp(['a squared discrepancy of ' num2str(sqdiscval,15)])
    %disp(['a stepsize of ' num2str(stepsize)])
    [xdes,sqdiscval,grad,stepsize] = ...
        optimizeDesign(kernelfun,xdes,sqdiscval,grad,stepsize,ntry,stopcrit);
    finaldisc(jj) = sqrt(sqdiscval); %best discrepancy found
    deschange(jj) = norm(xdes-xdesinit)/norm(xdesinit); %change in initial design
    discimprov(jj) = finaldisc(jj)/initdisc(jj); %change in initial design
%     nov2 = n;
    %disp('We end with a design and normalized gradient of')
    %disp([xdes,grad])
    %disp(['a squared discrepancy of ' num2str(sqdiscval,15)])
    %disp(['  for a reduction of ' num2str(1-sqdiscval/sqdiscvalinit(jj))])
    %disp(['a final stepsize of ' num2str(stepsize)])
end

figure(1)
loglog(nvec,finaldisc/finaldisc(1),'.',nvec,nvec(1)./nvec,'--')
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




