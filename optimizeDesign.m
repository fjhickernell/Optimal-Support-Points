%% Optimize the design for integration given the kernel function
function [xdes,sqdiscval,grad,stepsize] = ...
    optimizeDesign(kernelfun,xdes,sqdiscval,grad,stepsize,ntry,stopcrit)
stepmultiply = [0.5 1 2]; %different step size multipliers

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
        sqdiscimp = sqdiscval - sqdiscvalbest;
        xdes = xdestry(:,wh);
        stepsize = stepmultiply(wh)*stepsize;
        sqdiscval = sqdiscvalbest;
        if sqdiscimp/sqdiscval < stopcrit
            kk
            break; 
        else
            sqdiscimp/sqdiscval;
        end
        [~,grad] = kernelfun(xdes,[0,1]); %compute the new gradient
    else %we did not improve the design
        stepsize = 0.25*stepsize; %so take a smaller step
    end
end
stepsize*grad
