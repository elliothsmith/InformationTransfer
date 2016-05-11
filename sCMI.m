function [CMI,MI] = sCMI(Data1, Data2, Classification, tau)
% SCMI Symbolic conditional mutual information.
%   Determines the transfer information between Data1 and Data2 by
%   symbolically discretizing the signals and calculating the conditional 
%   mutual information between the two signals about the variable in
%   classification.
% 
%   The first output has two elements: 
%       1) information about the variable in Classification in Data1, given
%           that Data2 is known.
%       2) information about the variable in Classification in Data2, given
%           that Data1 is known.
%
%   The second output is mutual information between Data1 and Data2 
%       regardless of what is in Classification. 
%
%   Data variables should be M by N matrices where M is the number of
%       samples in each trial and N is the number of trials. Note: these are
%       single-trial analyses so N may be equal to 1.
%
%   Classification should be a N by 1 vector containing the trial type for
%       each trial.
%
%   tau is half of the size of the delay (units: number of time samples)
%       imposed on Data2. (a priori this could be zero for neural data, since
%       synaptic lags should be included in the recordings [author's bias]) The
%       few other CMI studies use delays ranging from several samples to
%       several milliseconds.
%
%
% Version Date: 2014102
% Author: Elliot H Smith

% matlabpool local 12

% number of signals to examine. Constrained to two for now. {i.e. I(classification;Data1|Data2)}
numChans = 2;

% constructing data tensor for further analyses.
if tau == 0 
    datamat(:,:,1) = Data1;
    datamat(:,:,2) = Data2;
else
    datamat(:,:,1) = [Data1 zeros(1,tau*2)];
    datamat(:,:,2) = [zeros(1,tau*2) Data2];
end

% number of trials in data matrices
numTrials =length(Classification);

%% symbolizing time series
% symbol size for discretizing the time series. Currently constrained to 3
% to manage complexity. This discretizes the time series into k! classes
% (6 classes for a symbol of length 3).
symbolSize = 3;
numClasses = factorial(symbolSize);

% adjusting data length to fit symbol size.
remz = mod(size(datamat,1),symbolSize);
if remz~=0
    datamat = datamat(1:end-remz,:,:);
end

% initializing clasification cell for discretized signal.
classesTrialsChans = cell(numTrials, numChans);
% display('Discretizing signals...')
for chs = 1:numChans % looping over channels
    % % update user regarding channels
    % display([' Channel ' num2str(chs) '/' num2str(numChans)])
    
    for tt = 1:numTrials % looping over trials
        % % update user regarding trials
        % if(mod(tt,500)==0 || tt==numTrials)
        %     disp(['  Trial ' num2str(tt) '/' num2str(numTrials)]);
        % end

        % getting discretized subvector for each trial.
        % length of the data divided by hte symbol size
        tiledTrial = reshape(datamat(:,tt,chs)',symbolSize,floor(size(datamat,1)./symbolSize));
        tiledTrial = tiledTrial';
        
        % determining symbol class for each trial. [20160304] I'm sure there is a more efficient/less gomey way of doing this. 
        tileClasses = zeros(size(tiledTrial,1),1);
        for tcl = 1:length(tileClasses)
            if tiledTrial(tcl,1)>tiledTrial(tcl,2)>tiledTrial(tcl,3) % (consecutive decrease) - weighted in king et al. 
                tileClasses(tcl) = 1;
            elseif tiledTrial(tcl,1)>tiledTrial(tcl,2) && tiledTrial(tcl,3)>tiledTrial(tcl,2) && tiledTrial(tcl,1)>tiledTrial(tcl,3) % (big decrease, small increase)
                tileClasses(tcl) = 2;
            elseif tiledTrial(tcl,1)<tiledTrial(tcl,2) && tiledTrial(tcl,2)>tiledTrial(tcl,3) && tiledTrial(tcl,1)>tiledTrial(tcl,3)% (small increase, big decrease)
                tileClasses(tcl) = 3;
            elseif tiledTrial(tcl,1)>tiledTrial(tcl,2) && tiledTrial(tcl,2)<tiledTrial(tcl,3) && tiledTrial(tcl,1)<tiledTrial(tcl,3) % (small decrease, big increase)
                tileClasses(tcl) = 4;
            elseif tiledTrial(tcl,3)>tiledTrial(tcl,2)>tiledTrial(tcl,1) % (consecutive increase) - weighted in king et al. 
                tileClasses(tcl) = 5;
            elseif tiledTrial(tcl,1)<tiledTrial(tcl,2) && tiledTrial(tcl,2)>tiledTrial(tcl,3) && tiledTrial(tcl,1)<tiledTrial(tcl,3) % (big increase, small decrease)
                tileClasses(tcl) = 6;
%             else % unclassified
%                 tileClasses(tcl) = 0;
            end
        end
        % saving tileClasses in a cell
        classesTrialsChans{tt,chs} = tileClasses;
    end
end

%% determining weighted symbolic mutual information between channels.
% generating frequency matrices for probabilities
XYmat1 = zeros(numClasses);
XYmat2 = zeros(numClasses);
XYmat12 = zeros(numClasses);
for trls = 1:numTrials % loooping over trials.
    tiles1 = cell2mat(classesTrialsChans(trls,1)); % discretized Channel 1
    tiles2 = cell2mat(classesTrialsChans(trls,2)); % discretized Channel 2
    for tls = 1:length(tiles1)
        
        %% [20160304] TODO: FIX THIS TO INCLUDE MORE CLASSES
        if Classification(trls)==2; % for first trial type (feedback)
            if tiles1(tls)~=0 && tiles2(tls)~=0
                XYmat1(tiles1(tls),tiles2(tls)) = XYmat1(tiles1(tls),tiles2(tls))+1;
            end
        elseif Classification(trls)==1 % for second trial type (nonfeedback)
            % [20160304] this part is commented out because it actually isn't required for CMI calculation. 
            %                 if tiles1(zz)~=0 && tiles2(zz)~=0
            %                     XYmat2(tiles1(zz),tiles2(zz)) = XYmat2(tiles1(zz),tiles2(zz))+1;
            %                 end
        end
        
        if tiles1(tls)~=0 && tiles2(tls)~=0
            XYmat12(tiles1(tls),tiles2(tls)) = XYmat12(tiles1(tls),tiles2(tls))+1;
        end        
    end
end

%% determining probability distributions. [20160304] definitely a more efficent way of doing this. 
ntrials12 = sum(sum(XYmat12));
ntrials1 = sum(sum(XYmat1));
Pf = sum(Classification==2)./numel(Classification);       % p(f) - uniform distribution => scalar multiply.
Pr2 = sum(XYmat12)./ntrials12;                            % p(r2)
Pr1 = sum(XYmat12,2)./ntrials12;                          % p(r1)
Pr1r2 = (XYmat12.*eye(6))./ntrials12;                     % p(r1,r2)
Pr1lr2 = Pr1r2 ./ repmat(Pr2,6,1);                        % p(r1|r2)
Pr2lr1 = Pr1r2 ./ repmat(Pr1,1,6);                        % p(r2|r1)
% in case of NaNs from divide by zeros
Pr1lr2(isnan(Pr1lr2))=0;
Pr2lr1(isnan(Pr2lr1))=0;

Pr2lf = sum(XYmat1)./ntrials1;                            % p(r1|f)
Pr1lf = sum(XYmat1,2)./ntrials1;                          % p(r2|f)
Pflr2 = (Pr2lf.*Pf)./Pr2;                                 % p(f|r2) - bayes' 
Pflr1 = (Pr1lf.*Pf)./Pr1;                                 % p(f|r1) - bayes' 
% in case of NaNs from divide by zeros
Pflr2(isnan(Pflr2))=0;
Pflr1(isnan(Pflr1))=0;

Pfr1lr2 = Pf .* Pr1lr2;                                   % p(f,r1|r2)
Pfr2lr1 = Pf .* Pr2lr1;                                   % p(f,r2|r1)

%% calculating conditional mutual information
% from area 1 to area 2
A = log2(Pfr1lr2./(repmat(Pflr2,6,1).*Pr1lr2));
A(isnan(A))=0;
A(isinf(A))=0;
Ifr1lr2 = sum(sum(repmat(Pr2,6,1) .* (Pfr1lr2 .* abs(A))));

% from area 2 to area 1
B = log2(Pfr2lr1./(repmat(Pflr1,1,6).*Pr2lr1));
B(isnan(B))=0;
B(isinf(B))=0;
Ifr2lr1 = sum(sum(repmat(Pr1,1,6) .* (Pfr2lr1 .* abs(B))));

% saving CMI results for each direction
CMI = [Ifr1lr2 Ifr2lr1];

%% calculating permutation information between the two channels. How many observations have you made? Is this robust? 
C = log2((Pr1r2)./(Pr1 * Pr2));
C(isinf(C)) = 0;
C(isnan(C)) = 0;
MI = sum(sum(Pr1r2 .* C));

end
