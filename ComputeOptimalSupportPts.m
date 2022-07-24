%% Compute Optimal Support Points

gail.InitializeWorkspaceDisplay

kernelfun = @RoughMatern;
ntry = 50; %number of gradient descent steps
stepmultiply = [0.5 1 2]; %different step size multipliers

nvec = 2.^(0:8);
nveclength = length(nvec);
finaldisc = zeros(nveclength,1);

for jj = 1:nveclength
    n = nvec(jj);
    xdestry = zeros(n,3);
    xdes = (1:n)'/n - 1/(2*n);
    xdesinit = xdes;
    [sqdiscval,grad] = kernelfun(xdes,[1,1]); %initial squared discrepancy and gradient
    sqdiscvalinit = sqdiscval;
    stepsize = 1/n;
    %disp('Beginning with a design of')
    %disp(xdes)
    %disp(['a squared discrepancy of ' num2str(sqdiscval,15)])
    %disp(['a stepsize of ' num2str(stepsize)])(

    for kk = 1:ntry
        nominalstep = stepsize*grad; %how far to move
        %compute new designs and their squared discrepancies
        xdestry(:,1) = xdes - stepmultiply(1)*nominalstep;
        sqdiscvaltry(1) = kernelfun(xdestry(:,1),[1,0]);
        xdestry(:,2) = xdes - stepmultiply(2)*nominalstep;
        sqdiscvaltry(2) = kernelfun(xdestry(:,2),[1,0]);
        xdestry(:,3) = xdes - stepmultiply(3)*nominalstep;
        sqdiscvaltry(3) = kernelfun(xdestry(:,3),[1,0]);
        [sqdiscvalbest,wh] = min(sqdiscvaltry); %find the best disrepancy
        if sqdiscvalbest < sqdiscval %we improved the design
            xdes = xdestry(:,wh);
            stepsize = stepmultiply(wh)*stepsize;
            sqdiscval = sqdiscvalbest;
        else %we did not improve the design
            stepsize = 0.25*stepsize; %so take a smaller step
        end
        [~,grad] = kernelfun(xdes,[0,1]); %compute the new gradient
    end
    finaldisc(jj) = sqrt(sqdiscval); %best discrepancy found
    deschange = norm(xdes-xdesinit);
    %disp('We end with a design and normalized gradient of')
    %disp([xdes,grad])
    %disp(['a squared discrepancy of ' num2str(sqdiscval,15)])
    %disp(['  for a reduction of ' num2str(1-sqdiscval/sqdiscvalinit)])
    %disp(['a final stepsize of ' num2str(stepsize)])
    disp(['relative l_2 design change ' num2str(deschange/norm(xdesinit))])
end


loglog(nvec,finaldisc/finaldisc(1),'.',nvec,nvec(1)./nvec,'--')
xlabel('\(n\)')
ylabel('Relative Discrepancy')


