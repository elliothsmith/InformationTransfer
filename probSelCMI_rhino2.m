%% [20160406] This code runs sCMI (or other kind of CMI?) on the data for the probilisitic reinforcement learning task as in Frank, Ramayya, and Zaghloul papers. 

matlabpool local 12

subDir =  '~/Data/ProbSel/';

datList = dir([subDir 'tensorized*.mat']);

%% [20160406] load first patient in order to get the task parameters here. 
load([subDir datList(1).name]);

% time periods for sliding window decode. (all in milliseconds) 
% [20160304] starting with a large window and then will scale down to test sensitivity.
durationMS = length(tmsec);
windowSize = 500;
stepSize = 10;
timePeriods = [0:stepSize:durationMS;(0:stepSize:durationMS)+windowSize];
timePeriods = timePeriods(:,timePeriods(2,:)<durationMS);

% looping over subjects for which I was able to successfully grab data..

% TODO: run for subjects 2 through the rest [20160503] 

listLen = length(datList);
for sb = 2:listLen

%% [20160406] load data here. 
load([subDir datList(sb).name]);

% Data is a 3D tensor: [events X samples X channels]
datamat = permute(Data,[2 1 3]);
% Data is now a 3D tensor: [samples X events X channels]

% only use thos subjects that have contact names. 
% It looks like a few subjects are missing contact names. 
% I may have already discarded those. 
if ~isempty(evDeets.leads)
		
	% finding all pairs of contacts...
	trodePairs = nchoosek(evDeets.leads,2);

	%% [20160503] initializing variables for parallelization. 
	CMI_FB = cell(size(timePeriods,2),size(trodePairs));
	MI_FB = zeros(size(timePeriods,2),size(trodePairs));
        shuffCMI_FB = cell(size(timePeriods,2),size(trodePairs));
        shuffMI_FB = zeros(size(timePeriods,2),size(trodePairs));

        CMI_S = cell(size(timePeriods,2),size(trodePairs));
        MI_S = zeros(size(timePeriods,2),size(trodePairs));
        shuffCMI_S = cell(size(timePeriods,2),size(trodePairs));
        shuffMI_S = zeros(size(timePeriods,2),size(trodePairs));

	%% for each channel pair, calculate CMI
	parfor prs = 1:size(trodePairs,1)
		display(sprintf('Calculating information transfer for subject number %d / %d for electrode combo %d / %d for Probabilistic Selection Task.',sb,listLen,prs,trodePairs));
 
		%% calulating sCMI for each time period
        	for tper = 1:size(timePeriods,2)
   			timePeriod = timePeriods(:,tper)';
   			timePeriod = int32((timePeriod)+1); % convert to sample indices
    		
    			% these are the data for sCMI calculation. They are shuffled below. 
    			Data1 = zscore(squeeze(datamat(timePeriod(1):timePeriod(2),fbIdx,trodePairs(prs,1))));
    			Data2 = zscore(squeeze(datamat(timePeriod(1):timePeriod(2),fbIdx,trodePairs(prs,2))));
    
    			% randomly shuffling the samples of the data in [Data1] and [Data2]
    			% not shuffling trials in order to keep the feedback structure the same. 
			% It remains to be seen whether this will affect CMI.
			% Theoretically it shouldn't,as the distributions of symbols are built up over trials.
			% shuffling trials should yield the same dstributions as unshuffled trials. 
			% Therefore shuffling saples is the way to go as that step will pertub probability distributions. [20160406]  
    			
			
			%% [20160407] adjusting feedback index to work as a class. 
			% fbClass = fbIdx+1;
			% This could be problematic: 
			% what we're really looking at is which contacts have informamtion abotu whether the event is a feedback event
			% rather than some other event and rather than looking at feedback identity. 
			% So, I need to include feedback identity in this.
			% I can probably find it in the events struct:: 
			FBevents = events(fbIdx);
			fbClass = double([FBevents.fb])+repmat(1,1,length(FBevents)); 
			
			% looping over feedback events.
		 	Ti = find(fbIdx);
    			shuffData1 = zeros(501,length(Ti)); shuffData2 = zeros(501,length(Ti)); 	
			for sh = 1:length(Ti)
    				timeBasis = timePeriod(1):timePeriod(2);
    				randIdx = randperm(length(timeBasis));
				jam = Ti(sh);
    				shuffData1(:,sh) = datamat(timeBasis(randIdx),jam,trodePairs(prs,1));
    				randIdx = randperm(length(timeBasis));
    				shuffData2(:,sh) = datamat(timeBasis(randIdx),jam,trodePairs(prs,2));
    			end
    
			%% [20160503] reshaping the shuffledData
			% shuffData1 = cell2mat(shuffData1);
			% shuffData2 = cell2mat(shuffData2);
			
    			%% [20160304] actually doing CMI.
			[CMI_FB{prs}(:,tper),~] = sCMI(Data1,Data2,fbClass,0);
			
			%% [20160304] running the same code on suffled data. 
    			[shuffCMI_FB{prs}(:,tper),~] = sCMI(shuffData1, shuffData2, fbClass,0);

			% clearing FB vars
			jam = [];
			Data1 = [];
			Data2 = [];
			shuffData1 = [];
			shuffData2 =[];

			%% [20160407] Now need to replicate the above lines of Code for stimulus events. 
			% I need to figure out what the stimulus classes are first.
			% I can also find this in the event struct: 
			Data1 = zscore(squeeze(datamat(timePeriod(1):timePeriod(2),stimIdx,trodePairs(prs,1))));
    			Data2 = zscore(squeeze(datamat(timePeriod(1):timePeriod(2),stimIdx,trodePairs(prs,2))));
			
			% indexing stimulus events. 
			Sevents = events(stimIdx);
			sClass = double([Sevents.pairNum])+repmat(1,1,length(Sevents)); 
			
			% looping over feedback events. 
			Tj = find(stimIdx);
    			shuffData1 = zeros(501,length(Ti)); shuffData2 = zeros(501,length(Ti)); 	
			for si = 1:length(Tj)
    				timeBasis = timePeriod(1):timePeriod(2);
    				randIdx = randperm(length(timeBasis));
    				jam = Tj(si);
				shuffData1(:,si) = datamat(timeBasis(randIdx),jam,trodePairs(prs,1));
    				randIdx = randperm(length(timeBasis));
    				shuffData2(:,si) = datamat(timeBasis(randIdx),jam,trodePairs(prs,2));
    			end
    			
                        %% [20160503] reshaping the shuffledData
                        % shuffData1 = cell2mat(shuffData1);
                        % shuffData2 = cell2mat(shuffData2);
	
    			%% [20160304] actually doing CMI.
			[CMI_S{prs}(:,tper),~] = sCMI(Data1,Data2,sClass,0);
			
			%% [20160304] running the same code on suffled data. 
    			[shuffCMI_S{prs}(:,tper),~] = sCMI(shuffData1, shuffData2, sClass,0);

			%% [20160407] I now have enough to make a cool video showing symbolic information transfer in the brain. very cool. 
			% TODO stuff that is even cooler:
			% TODO update CMI in a bayesian manner with reward probabilities. Super cool. 
			% TODO First decode feedback and stimulus and then do CMI for decode probability distributions. 
			% 	- Does this approach improve the representations? 
			% 	- how does it affeect CMI measurement?  
			% TODO put CMI in Q-learning framework. 
			% 	- how much variance is explained? 
			

    		end % looping over time periods. 
	end % looping over pairwise combos of trodes

	%% [20160407] I'll also do hypohtesis testing in the visualization script. 
	

	%% save data for visualization with other scripts...
	saveStr = sprintf('%ssymbolicCMI_forProbabilisticSelectionTask_subject%s.mat',subDir,events(1).subject);
	save(saveStr,'timePeriods','trodePairs','stepSize','windowSize','durationMS','CMI_FB','MI_FB','shuffCMI_FB','shuffMI_FB','fbClass','FBevents','CMI_S','MI_S','shuffCMI_S','shuffMI_S','sClass','Sevents','-v7.3');

	% clearvars -except sb subDir datList listLen durationMS windowSize stepSize 

else % no leeds for the current patient. 
	display(sprintf('event leads are empty for patient %s',events(1).subject))
end

end % looping over subjects. 



