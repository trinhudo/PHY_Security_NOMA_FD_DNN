clc; clear; close all;

% The number of transmitter
MM = 4;
% Transmit power at transmitter
PS_dB = -20:1:60;
PS = 10.^(PS_dB./10);
%
PN_dB = 10;
PN = 10^(PN_dB/10);
%
PF_dB = 10;
PF = 10^(PF_dB/10);
%
PE_dB = 5;
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
Rth_xN = 0.1;
gth_xN = 2^(Rth_xN);
%
Rth_xF = 0.1;
gth_xF = 2^(Rth_xF);
%
thetaN = 0.01:0.01:0.5;
thetaF = 1 - (thetaN);
%
beta = 0.1;
%
SimTimes = 1e5;

for ss = 1:length(PS_dB)
    for aa = 1:length(thetaN)
        fprintf('Simulation is running: thetaN = %d \n ', thetaN(aa))
        for bb = 1:length(beta)
            fprintf('beta = %d  \n', beta(bb) )
            % channel modeling
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
            %
            snrSiN_xN = PS(ss).*thetaN(aa).*giN./(PS(ss).*beta(bb).*thetaF(aa).*giN + PE.*gEN + 2);
            snrSiN_xF = PS(ss).*thetaF(aa).*giN./(PS(ss).*thetaN(aa).*giN + PE.*gEN + 2);
            %
            snrSiF_xF = PS(ss).*thetaF(aa).*giF./(PS(ss).*thetaN(aa).*giF + PE.*gEF + 2);
            %
            snrSiE_xN = PS(ss).*thetaN(aa).*giE./(PN.*gNE + PF.*gFE + 2);
            snrSiE_xF = PS(ss).*thetaF(aa).*giE./(PN.*gNE + PF.*gFE + 2);
            
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
            %
            CSbN_xF = log2(1 + snrSbN_xF);
            %
            Cs_xN = max( log2(1+ snrSbN_xN) - log2(1 + snrSbE_xN), 0);
            Cs_xF = max( log2(1+ snrSbF_xF) - log2(1 + snrSbE_xF), 0);
            %
            %% Simulation of SOP
            sim_sop_xN = 0;
            sim_sop_xF = 0;
            %
            for yy = 1:SimTimes
                % Near User
                if(Cs_xN(yy) < Rth_xN && CSbN_xF(yy) >= Rth_xF)
                    sim_sop_xN = sim_sop_xN + 1;
                end
                if(CSbN_xF(yy) < Rth_xF)
                    sim_sop_xN = sim_sop_xN + 1;
                end
                
                % Far User
                if(Cs_xF(yy) < Rth_xF)
                    sim_sop_xF = sim_sop_xF + 1;
                end
            end
            %
            SOP_xN(ss, aa) = sim_sop_xN/SimTimes;
            SOP_xF(ss, aa) = sim_sop_xF/SimTimes;
            %
            TSOP(ss, aa) = 1 - (1 - SOP_xN(ss, aa))*(1 - SOP_xF(ss, aa));
        end
    end
end
%
[idx_PS,idx_thetaN]=find(TSOP==min(TSOP(:)));
opt_thetaN = thetaN(idx_thetaN);
opt_PS = PS(idx_PS);
opt_PS_dB = PS_dB(idx_PS);
min_SystemSOP = TSOP(idx_PS, idx_thetaN);
%
[X Y] = meshgrid(thetaN, PS_dB);
%
figure(3)
surf(thetaN, PS_dB, TSOP, 'linestyle', '-')
hold on
plot3(opt_thetaN, PS_dB(idx_PS), min_SystemSOP,'.r','markersize',20);

save test_PS_thetaN_SystemSOP_OTS.dat TSOP -ascii


% set(gca, 'ZScale', 'log')
% set(gca,'XTick',0:.5:1)
% set(gca,'YTick',0:.5:1)
% %
% xlabel('\rho')
% ylabel('\alpha')
% zlabel('Outage Probability')
% legend('NearUser', 'FarUser', 'System')

