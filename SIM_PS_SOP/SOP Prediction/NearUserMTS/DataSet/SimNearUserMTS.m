% function for the cell-center user's SOP with RTS scheme
function output = SimNearUserMTS(MM, PS_dB, PN_dB, PF_dB, PE_dB, dSN, dSF, dSE, dNE, dFE, thetaN, beta, Rth_xN, Rth_xF, SimTimes )
%% Theory Model of SOP with MTS scheme
%
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
%% Simulation Part
%Channel model
for mm = 1:MM
    hiN(:, mm) = random('Rayleigh', sqrt(lSN/2), [1, SimTimes]);
    hiF(:, mm) = random('Rayleigh', sqrt(lSF/2), [1, SimTimes]);
    hiE(:, mm) = random('Rayleigh', sqrt(lSE/2), [1, SimTimes]);
end
hNE(:, 1) = random('Rayleigh', sqrt(lNE/2), [1, SimTimes]);
hEN(:, 1) = random('Rayleigh', sqrt(lEN/2), [1, SimTimes]);
%
hFE(:, 1) = random('Rayleigh', sqrt(lFE/2), [1, SimTimes]);
hEF(:, 1) = random('Rayleigh', sqrt(lEF/2), [1, SimTimes]);
%
giN = abs(hiN).^2;
giF = abs(hiF).^2;
giE = abs(hiE).^2;
%
gNE = abs(hNE).^2;
gEN = abs(hEN).^2;
%
gFE = abs(hFE).^2;
gEF = abs(hEF).^2;
% MTS scheme
for yy = 1:SimTimes
    [gSE MTS] = min(giE, [], 2);
    %
    gSN(yy, 1) = giN(yy, MTS(yy));
    gSF(yy, 1) = giF(yy, MTS(yy));
end
snrSbN_xN = PS*thetaN*gSN./(PS*beta*thetaF*gSN + PE*gEN + 2);
snrSbN_xF = PS*thetaF*gSN./(PS*thetaN*gSN + PE*gEN + 2);
%
snrSbF_xF = PS*thetaF*gSF./(PS*thetaN*gSF + PE*gEF + 2);
%
snrSbE_xN = PS*thetaN*gSE./(PN*gNE + PF*gFE + 2);
snrSbE_xF = PS*thetaF*gSE./(PN*gNE + PF*gFE + 2);
%
CSbN_xF = log2(1 + snrSbN_xF);
%
Cs_xN = max( log2(1+ snrSbN_xN) - log2(1 + snrSbE_xN), 0);
Cs_xF = max( log2(1+ snrSbF_xF) - log2(1 + snrSbE_xF), 0);

%% Simulation of SOP
sim_psiA = 0;
sim_psiB = 0;
for yy = 1:SimTimes
    % SIM: PsiA
    if(Cs_xN(yy) < Rth_xN && CSbN_xF(yy) >= Rth_xF)
        sim_psiA = sim_psiA + 1;
    end
    % SIM: PsiB
    if(CSbN_xF(yy) < Rth_xF)
        sim_psiB = sim_psiB + 1;
    end
end
%
SIM_PsiA = sim_psiA/SimTimes;
SIM_PsiB = sim_psiB/SimTimes;
%
output = SIM_PsiA + SIM_PsiB;
end

