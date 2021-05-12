%
tic
clear all
clc

% The number of sample data for training
SampleNumber = 4e3;
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
SimTimes = 5e4;
step = 1e3;
%
data_set = [];
for ii = 1:SampleNumber
    if(mod(ii, 5e2) == 0)
        fprintf('Running %d over %d \n', ii, SampleNumber)
    end
    SOP(ii,1) = SimNearUserOTS(MM(ii), PS_dB(ii), PN_dB(ii), PF_dB(ii), PE_dB(ii), dSN(ii), dSF(ii), dSE(ii), dNE(ii), dFE(ii), thetaN(ii), beta(ii), Rth_xN(ii), Rth_xF(ii), SimTimes);
    
    data_set = [data_set; MM(ii), PS_dB(ii), PN_dB(ii), PF_dB(ii), PE_dB(ii), dSN(ii), dSE(ii), dNE(ii), dFE(ii), thetaN(ii), Rth_xN(ii), Rth_xF(ii), SOP(ii)];

    if mod(ii,step)==0
        fprintf('save %d over %d \n', ii, SampleNumber)
        filename = ['DataSet_NearUserOTS_',num2str(ii),'.csv'] ;
        csvwrite(filename, data_set)
        clear data_set
        data_set = [];
    end
end

all_csv = [];
for kk = step:step:SampleNumber
    filename = ['DataSet_NearUserOTS_',num2str(kk),'.csv'] ;
    all_csv = [all_csv; csvread(filename)]; % Concatenate vertically
end
csvwrite('DataSet_NearUserOTS_1e5.csv', all_csv);

aa = toc;
ss=seconds(aa);
ss.Format = 'hh:mm:ss.SSS'

