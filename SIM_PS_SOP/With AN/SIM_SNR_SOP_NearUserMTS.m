% CODE: cell-center user's SOP with MTS scheme
clear; clc; close all;

% The number of transmitter
MM = 4;
% Transmit power at transmitter
SNR_dB = -20:1:60;
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
SimTimes = 1e0;
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
    % MTS scheme
    for yy = 1:SimTimes
        [gSE MTS] = min(giE, [], 2);
        %
        gSN(yy, 1) = giN(yy, MTS(yy));
        gSF(yy, 1) = giF(yy, MTS(yy));
    end
    snrSbN_xN = SNR(ss)*thetaN*gSN./(SNR(ss)*beta*thetaF*gSN + PE*gEN + 2);
    snrSbN_xF = SNR(ss)*thetaF*gSN./(SNR(ss)*thetaN*gSN + PE*gEN + 2);
    %
    snrSbF_xF = SNR(ss)*thetaF*gSF./(SNR(ss)*thetaN*gSF + PE*gEF + 2);
    %
    snrSbE_xN = SNR(ss)*thetaN*gSE./(PN*gNE + PF*gFE + 2);
    snrSbE_xF = SNR(ss)*thetaF*gSE./(PN*gNE + PF*gFE + 2);
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
    SIM_PsiA(ss) = sim_psiA/SimTimes;
    SIM_PsiB(ss) = sim_psiB/SimTimes;
    %
    SIM_SOP_xN(ss) = SIM_PsiA(ss) + SIM_PsiB(ss);
    
    %% Analysis
    %
    a1 = gth_xN*SNR(ss)*thetaN*lSE;
    a2 = MM*PN*lNE;
    a3 = MM*PF*lFE;
    omega3 =  thetaN/beta/thetaF - (gth_xN -1);
    test3 = 0;
    %
    for rr = 1:RR
        xr = cos((2*rr-1)*pi/RR);
        xx = omega3*xr/2 + omega3/2;
        %
        test3 = test3 + omega3*pi/2/RR *sqrt(1 - xr^2) *SNR(ss)*(thetaN - beta*thetaF*(xx + (gth_xN-1)))*lSN * ...
            exp(-2*(xx + (gth_xN - 1))/SNR(ss)/(thetaN - beta*thetaF*(xx + (gth_xN-1)))/lSN) / ...
            (PE*lEN*(xx + (gth_xN-1)) + SNR(ss)*(thetaN - beta*thetaF*(xx + (gth_xN-1)))*lSN) * ...
            (a1*a2/(a2*xx+ a1) + a1*a3/(a3*xx + a1) + 2*MM)*a1*exp(-2*MM*xx/a1)/(a2*xx+a1)/(a3*xx+a1);
    end
    ANA_PsiA1(ss) = 1 - test3;
    % PsiA2
    ANA_PsiA2(ss) = 1 - SNR(ss)*(thetaF - thetaN*(gth_xF-1))*lSN *exp(- 2*(gth_xF -1)/SNR(ss)/(thetaF - thetaN*(gth_xF - 1))/lSN) / ...
        ((gth_xF-1)*PE*lEN + SNR(ss)*(thetaF - thetaN*(gth_xF -1))*lSN);
    ANA_PsiA(ss) = ANA_PsiA1(ss) - ANA_PsiA2(ss);
    
    % PsiB
    ANA_PsiB(ss) = 1 - SNR(ss)*(thetaF - thetaN*(gth_xF-1))*lSN *exp(- 2*(gth_xF -1)/SNR(ss)/(thetaF - thetaN*(gth_xF - 1))/lSN) / ...
        ((gth_xF-1)*PE*lEN + SNR(ss)*(thetaF - thetaN*(gth_xF -1))*lSN);
    %
    ANA_SOP_xN(ss) = ANA_PsiA(ss) + ANA_PsiB(ss);
end
%
semilogy(SNR_dB, SIM_SOP_xN, '-r')
hold on
semilogy(SNR_dB, ANA_SOP_xN, 'rs')


% save data_SNR_SOP_NearUserMTS_sim.dat SIM_SOP_xN -ascii
save data_SNR_SOP_NearUserMTS_ana.dat ANA_SOP_xN -ascii

