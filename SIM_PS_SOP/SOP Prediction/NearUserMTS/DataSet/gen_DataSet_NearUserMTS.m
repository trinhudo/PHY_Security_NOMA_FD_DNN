
% generate data sample without throughput
% @author: van_nguyentoan
%
tic
clear all
clc

% The number of sample data for training
SampleNumber = 1e5;
% The number of transmitter

MM = 4* ones(SampleNumber, 1); % 4
%
PS_dB = -20 + 5* randi([0 16], SampleNumber, 1);
PN_dB = 5*randi([1 2], SampleNumber, 1);
PF_dB = 5*randi([1 2], SampleNumber, 1);
PE_dB = 5*randi([1 2], SampleNumber, 1);
%
dSN = 0.2*randi([1 2], SampleNumber, 1);
dSF = ones(SampleNumber, 1);
dSE = 0.5*randi([1 2], SampleNumber, 1);
%
dNE = 0.5*randi([1 2], SampleNumber, 1);
dFE = 0.5*randi([1 2], SampleNumber, 1);
%
thetaN = 0.1*randi([1 2], SampleNumber, 1);
beta = 0.1*ones(SampleNumber, 1);
%
Rth_xN = 0.1*randi([1 2], SampleNumber, 1);
Rth_xF = 0.1*randi([1 2], SampleNumber, 1);
%
SimTimes = 2e4;
%
for ii = 1:SampleNumber
    if(mod(ii, 1e3) == 0)
        fprintf('Running %d over %d \n', ii, SampleNumber)
    end
    SOP(ii,1) = SimNearUserMTS(MM(ii), PS_dB(ii), PN_dB(ii), PF_dB(ii), PE_dB(ii), dSN(ii), dSF(ii), dSE(ii), dNE(ii), dFE(ii), thetaN(ii), beta(ii), Rth_xN(ii), Rth_xF(ii), SimTimes);
end
SampleData = [MM, PS_dB, PN_dB, PF_dB, PE_dB, dSN, dSE, dNE, dFE, thetaN, Rth_xN, Rth_xF, SOP];

csvwrite('DataSet_NearUserMTS_Kyusung_1e5.csv',SampleData);
aa = toc;
ss=seconds(aa);
ss.Format = 'hh:mm:ss.SSS'
%
% % data = readmatrix('datasetMay19.csv','Range',3); % starting from line 3
%
%






