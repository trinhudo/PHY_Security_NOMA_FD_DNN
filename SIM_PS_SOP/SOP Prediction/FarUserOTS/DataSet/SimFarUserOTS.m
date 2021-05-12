% CODE: cell-center user's SOP with MTS scheme
function out = SimFarUserOTS2(MM, PS_dB, PN_dB, PF_dB, PE_dB, dSN, dSF, dSE, dNE, dFE, thetaN, beta, Rth_xN, Rth_xF, SimTimes)
% The number of transmitter

PS = 10.^(PS_dB./10);
PN = 10^(PN_dB/10);
PF = 10^(PF_dB/10);
PE =10^(PE_dB/10);
% Channel Envoirments
epsilon = 2.7;
L = 1e3;
d0 = 1;
%
dEN = dNE;
dEF = dFE;
%
lSN = L/(dSN/d0)^epsilon;
lSF = L/(dSF/d0)^epsilon;
lSE = L/(dSE/d0)^epsilon;
%
lNE = L/(dNE/d0)^epsilon;
lEN = L/(dEN/d0)^epsilon;
%
lFE = L/(dFE/d0)^epsilon;
lEF = L/(dEF/d0)^epsilon;
%
thetaF = 1 - (thetaN);
%
gth_xN = 2^(Rth_xN);
gth_xF = 2^(Rth_xF);

    
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
    [CT OTS] = max( (CSiN_xN -CSiE_xN)+(CSiF_xF - CSiE_xF), [], 2);
    for ss = 1:SimTimes
          snrSbN_xN(ss, 1) = snrSiN_xN(ss, OTS(ss));
        snrSbN_xF(ss, 1) = snrSiN_xF(ss, OTS(ss));
        %
        snrSbF_xF(ss, 1) = snrSiF_xF(ss, OTS(ss));
        %
        snrSbE_xN(ss, 1) = snrSiE_xN(ss, OTS(ss));
        snrSbE_xF(ss, 1) = snrSiE_xF(ss, OTS(ss));
    end
    %
     CSbN_xF = log2(1 + snrSbN_xF);
    %
    Cs_xN = max( log2(1+ snrSbN_xN) - log2(1 + snrSbE_xN), 0);
    Cs_xF = max( log2(1+ snrSbF_xF) - log2(1 + snrSbE_xF), 0);
    
    %% Simulation of SOP
    sim_SOP_xF = 0;
    for ss = 1:SimTimes
        if(Cs_xF(ss) < Rth_xF)
            sim_SOP_xF = sim_SOP_xF + 1;
        end
    end
    out = sim_SOP_xF/SimTimes;
end