% function for the User N with OTS scheme
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

snrSiN_xN = PS*thetaN*giN./(PS*beta*thetaF*giN + PE*gEN + 2);
snrSiN_xF = PS*thetaF*giN./(PS*thetaN*giN + PE*gEN + 2);
%
snrSiF_xF = PS*thetaF*giF./(PS*thetaN*giF + PE*gEF + 2);
%
snrSiE_xN = PS*thetaN*giE./(PN*gNE + PF*gFE + 2);
snrSiE_xF = PS*thetaF*giE./(PN*gNE + PF*gFE + 2);

CSiN_xN = log2(1 + snrSiN_xN);
CSiN_xF = log2(1 + snrSiN_xF);
%
CSiF_xF = log2(1 + snrSiF_xF);
%
CSiE_xN = log2(1 + snrSiE_xN);
CSiE_xF = log2(1 + snrSiE_xF);
%
for yy = 1:SimTimes
   [CT OTS] = max( (CSiN_xN -CSiE_xN)+(CSiF_xF - CSiE_xF), [], 2); 
   %
        snrSbN_xN(yy, 1) = snrSiN_xN(yy, OTS(yy));
        snrSbN_xF(yy, 1) = snrSiN_xF(yy, OTS(yy));
        %
        snrSbF_xF(yy, 1) = snrSiF_xF(yy, OTS(yy));
        %
        snrSbE_xN(yy, 1) = snrSiE_xN(yy, OTS(yy));
        snrSbE_xF(yy, 1) = snrSiE_xF(yy, OTS(yy));
end

    Cs_xN = max(log2(1 + snrSbN_xN) - log2(1 + snrSbE_xN), 0);
    Cs_xF = max(log2(1 + snrSbN_xF) - log2(1 + snrSbE_xF), 0);
    %
    CSbN_xF = log2(1 + snrSbN_xF);

%% Simulation of SOP
sim_nearuserA = 0;
sim_nearuserB = 0;
for yy = 1:SimTimes
    % SIM: PsiA
    if(Cs_xN(yy) < Rth_xN && CSbN_xF(yy) >= Rth_xF)
        sim_nearuserA = sim_nearuserA + 1;
    end
    % SIM: PsiB
    if(CSbN_xF(yy) < Rth_xF)
        sim_nearuserB = sim_nearuserB + 1;
    end
end
%
SIM_NearUserA = sim_nearuserA/SimTimes;
SIM_NearUserB = sim_nearuserB/SimTimes;
%
output = SIM_NearUserA + SIM_NearUserB;
end

