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
PE_dB = 0:10:60;
PE =10.^(PE_dB./10);
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
thetaN = 0.01:0.03:0.5;
thetaF = 1 - (thetaN);
%
beta = 0.1;
%
SimTimes = 5e4;

for xx = 1:length(PE_dB)
    fprintf('Running Simulation: PE = %d (dB) \n', PE_dB(xx))
    for yy = 1:length(PS_dB)
        for zz = 1:length(thetaN)
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
            for ss = 1:SimTimes
                RTS = randperm(MM, 1);
                %
                gSN(ss, 1) = giN(ss, RTS);
                gSF(ss, 1) = giF(ss, RTS);
                gSE(ss, 1) = giE(ss, RTS);
            end
            %
            snrSbN_xN = PS(yy)*thetaN(zz)*gSN./(PS(yy)*beta*thetaF(zz)*gSN + PE(xx)*gEN + 1);
            snrSbN_xF = PS(yy)*thetaF(zz)*gSN./(PS(yy)*thetaN(zz)*gSN + PE(xx)*gEN + 1);
            %
            snrSbF_xF = PS(yy)*thetaF(zz)*gSF./(PS(yy)*thetaN(zz)*gSF + PE(xx)*gEF + 1);
            %
            snrSbE_xN = PS(yy)*thetaN(zz)*gSE./(2);
            snrSbE_xF = PS(yy)*thetaF(zz)*gSE./(2);
            %
            CSbN_xF = log2(1 + snrSbN_xF);
            %
            Cs_xN = max( log2(1+ snrSbN_xN) - log2(1 + snrSbE_xN), 0);
            Cs_xF = max( log2(1+ snrSbF_xF) - log2(1 + snrSbE_xF), 0);
            
            %% Simulation Result of SOP
            sim_SOP_xN = 0;
            sim_SOP_xF = 0;
            for ss = 1:SimTimes
                % with AN
                if (Cs_xN(ss) < Rth_xN && CSbN_xF(ss) > Rth_xF)
                    sim_SOP_xN = sim_SOP_xN + 1;
                end
                if(CSbN_xF(ss) < Rth_xF)
                    sim_SOP_xN = sim_SOP_xN + 1;
                end
                %
                if(Cs_xF(ss) < Rth_xF)
                    sim_SOP_xF = sim_SOP_xF + 1;
                end
            end
            %
            SIM_SOP_xN(yy, zz) = sim_SOP_xN/SimTimes;
            SIM_SOP_xF(yy, zz) = sim_SOP_xF/SimTimes;
            TSOP(yy, zz) = 1 - (1 - SIM_SOP_xN(yy,zz))*(1 - SIM_SOP_xF(yy,zz));
        end
    end
    % find the optimal PS and thetaN
    [idx_PS, idx_thetaN] = find(TSOP == min(TSOP(:)));
    opt_thetaN = thetaN(idx_thetaN);
    opt_PS = PS(idx_PS);
    opt_PS_dB = PS_dB(idx_PS);
    % save the system SOP
    min_SystemSOP(xx) = TSOP(idx_PS, idx_thetaN);
    fprintf('Fing the optimal point (%f, %f) when PE is %d\n', opt_thetaN, opt_PS_dB, PE_dB(xx))
end

semilogy(PE_dB, min_SystemSOP, '--ks')
xlabel({'$\bar{\gamma}_{\mathsf{S}}$'},'Interpreter','latex')
ylabel('System SOP')

save data_systemSOP_PE_RTS_wo_AN.dat min_SystemSOP -ascii