% function for the cell-center user's SOP with RTS scheme
function output = SimFarUserRTS(MM, PS_dB, PN_dB, PF_dB, PE_dB, dSN, dSF, dSE, dNE, dFE, thetaN, Rth_xN, Rth_xF, SimTimes)
%% Theory Model of SOP with RTS scheme
PS = 10.^(PS_dB./10);
PN = 10.^(PN_dB./10);
PF = 10.^(PF_dB./10);
PE = 10.^(PE_dB./10);
%
epsilon = 2.7;
L = 1e3;
d0 = 1;
%
lSN = L/(dSN/d0)^epsilon;
lSF = L/(dSF/d0)^epsilon;
lSE = L/(dSE/d0)^epsilon;
lNE = L/(dNE/d0)^epsilon;
lFE = L/(dFE/d0)^epsilon;
lEN = lNE;
lEF = lFE;
% Fixed parameters
thetaF = 1 - thetaN;
%
gth_xN = 2^Rth_xN;
gth_xF = 2^Rth_xF;

for mm = 1:MM
    hiF(:, mm) = random('Rayleigh', sqrt(lSF/2), [1, SimTimes]);
    hiE(:, mm) = random('Rayleigh', sqrt(lSE/2), [1, SimTimes]);
end
hNE(:, 1) = random('Rayleigh', sqrt(lNE/2), [1, SimTimes]);
hEN(:, 1) = random('Rayleigh', sqrt(lEN/2), [1, SimTimes]);
%
hFE(:, 1) = random('Rayleigh', sqrt(lFE/2), [1, SimTimes]);
hEF(:, 1) = random('Rayleigh', sqrt(lEF/2), [1, SimTimes]);
%
giF = abs(hiF).^2;
giE = abs(hiE).^2;
%
gNE = abs(hNE).^2;
gEN = abs(hEN).^2;
%
gFE = abs(hFE).^2;
gEF = abs(hEF).^2;
% RTS scheme
for yy = 1:SimTimes
    RTS = randperm(MM, 1);
    %
    gSF(yy, 1) = giF(yy, RTS);
    gSE(yy, 1) = giE(yy, RTS);
end
%
snrSbF_xF = PS*thetaF*gSF./(PS*thetaN*gSF + PE*gEF + 2);
%
snrSbE_xF = PS*thetaF*gSE./(PN*gNE + PF*gFE + 2);
%
Cs_xF = max( log2(1+ snrSbF_xF) - log2(1 + snrSbE_xF), 0);
%
cnt = Cs_xF < Rth_xF;
output = sum(cnt)/SimTimes;
end

