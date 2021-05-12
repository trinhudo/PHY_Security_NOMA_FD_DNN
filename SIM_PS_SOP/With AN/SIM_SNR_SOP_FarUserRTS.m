% CODE: cell-edge user's SOP with RTS scheme
%clear; clc; close all;

% The number of transmitter
MM = 4;
% Transmit power at transmitter
SNR_dB = -20:10:60;
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
SimTimes = 1e5;
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
    % RTS scheme
    for yy = 1:SimTimes
        RTS = randperm(MM, 1);
        %
        gSN(yy, 1) = giN(yy, RTS);
        gSF(yy, 1) = giF(yy, RTS);
        gSE(yy, 1) = giE(yy, RTS);
    end
    %
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
    
    %% Simulation Result of SOP
    SS = gth_xN*SNR(ss)*thetaF*gSE./(PN*gNE + PF*gFE + 2);
    %
    sim_SOP_xF = 0;
    tempA1 = 0;
    for zz = 1:SimTimes
        % definition of PsiA1 in (6)
        if(Cs_xF(zz) < Rth_xF)
            sim_SOP_xF = sim_SOP_xF + 1;
        end
    end
    SIM_SOP_xF(ss) = sim_SOP_xF/SimTimes;
    
    %% Analysis of 
    % eq 19.
    a1 = gth_xF*SNR(ss)*thetaF*lSE;
    a2 = PN*lNE;
    a3 = PF*lFE;
    %
    omega2 =  thetaF/thetaN - (gth_xF -1);
    % After Quadrature
    test2 = 0;
    for rr = 1:RR
        xr = cos((2*rr-1)*pi/RR);
        xx = omega2*xr/2 + omega2/2;
        %
        test2 = test2 + omega2*pi/2/RR *sqrt(1 - xr^2) *SNR(ss)*(thetaF - thetaN*(xx + (gth_xF-1)))*lSF * ...
            exp(-2*(xx+(gth_xF - 1))/SNR(ss)/(thetaF - thetaN*(xx + (gth_xF-1)))/lSF) / ...
            (PE*lEF*(xx + (gth_xF-1)) + SNR(ss)*(thetaF - thetaN*(xx + (gth_xF-1)))*lSF) * ...
            (a1*a2/(a2*xx+ a1) + a1*a3/(a3*xx + a1) + 2)*a1*exp(-2*xx/a1)/(a2*xx+a1)/(a3*xx+a1);
    end
    Approx_RTS_SOP_xF(ss) = 1 - test2;
    ANA_SOP_xF(ss) = Approx_RTS_SOP_xF(ss);
end
%
figure(1) %PsiA
semilogy(SNR_dB, SIM_SOP_xF, '-b')
hold on
semilogy(SNR_dB, ANA_SOP_xF, 'b>')

% % save data_SNR_SOP_FarUserRTS_sim.dat SIM_SOP_xF -ascii
% save data_SNR_SOP_FarUserRTS_ana.dat ANA_SOP_xF -ascii

