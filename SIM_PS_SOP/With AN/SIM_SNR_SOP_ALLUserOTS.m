% CODE: cell-center user's SOP with MTS scheme
clear; clc; close all;

% The number of transmitter
MM = 4;
% Transmit power at transmitter
SNR_dB = -20:5:60;
SNR = 10.^(SNR_dB./10);
%
PN_dB = 10;
PN = 10^(PN_dB/10);
%
PF_dB = 10;
PF = 10^(PF_dB/10);
%
PE_dB = 10;
PE =10^(PE_dB/10);
% Channel Envoirments
epsilon = 2.7;
L = 1e3;
d0 = 1;
%
dSN = 0.2;
dSF = 1;
dSE = 1;
%
dNE = 0.5;
dEN = dNE;
%
dFE = 0.5;
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
Rth_xN = 0.2;
gth_xN = 2^(Rth_xN);
%
Rth_xF = 0.2;
gth_xF = 2^(Rth_xF);
%
thetaN = 0.2;
thetaF = 1 - (thetaN);
%
beta = 0.1;
%
SimTimes = 2e4;
%
RR = 1e5;

for ss = 1:length(SNR_dB)
    fprintf('Running Simulation: SNR = %d (dB) \n', SNR_dB(ss))
    
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
    
    snrSiN_xN = SNR(ss)*thetaN*giN./(SNR(ss)*beta*thetaF*giN + PE*gEN + 2);
    snrSiN_xF = SNR(ss)*thetaF*giN./(SNR(ss)*thetaN*giN + PE*gEN + 2);
    %
    snrSiF_xF = SNR(ss)*thetaF*giF./(SNR(ss)*thetaN*giF + PE*gEF + 2);
    %
    snrSiE_xN = SNR(ss)*thetaN*giE./(PN*gNE + PF*gFE + 2);
    snrSiE_xF = SNR(ss)*thetaF*giE./(PN*gNE + PF*gFE + 2);
    
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
    CSbN_xF = log2(1 + snrSbN_xF);
    %
    Cs_xN = max( log2(1+ snrSbN_xN) - log2(1 + snrSbE_xN), 0);
    Cs_xF = max( log2(1+ snrSbF_xF) - log2(1 + snrSbE_xF), 0);
    
    %% Simulation of SOP
    sim_SOP_xN = 0;
    for yy = 1:SimTimes
        % SIM: PsiA
        if(Cs_xN(yy) < Rth_xN && CSbN_xF(yy) >= Rth_xF)
            sim_SOP_xN = sim_SOP_xN + 1;
        end
        % SIM: PsiB
        if(CSbN_xF(yy) < Rth_xF)
            sim_SOP_xN = sim_SOP_xN + 1;
        end
    end
    SIM_SOP_xN(ss) = sim_SOP_xN/SimTimes;

    %% Far User SOP
    sim_SOP_xF = 0;
    for zz = 1:SimTimes
        if(Cs_xF(zz) < Rth_xF)
            sim_SOP_xF = sim_SOP_xF + 1;
        end
    end
    SIM_SOP_xF(ss) = sim_SOP_xF/SimTimes;
    
    
    SIM_TSOP(ss) = 1 - (1 - SIM_SOP_xN(ss))*(1 - SIM_SOP_xF(ss));
end
%
semilogy(SNR_dB, SIM_SOP_xN, '-r')
hold on
semilogy(SNR_dB, SIM_SOP_xF, '-b')
%
% save test_SNR_SOP_NearUserOTS_2sim.dat SIM_SOP_xN -ascii
% save test_SNR_SOP_FarUserOTS_2sim.dat SIM_SOP_xF -ascii


