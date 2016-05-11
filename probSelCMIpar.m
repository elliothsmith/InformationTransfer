%% This code implements the UPenn & iEEG.org functions in order to run sCMI (or other kind of CMI?) on the data for the probilisitic reinforcement learning task as in Frank, Ramayya, and Zaghloul papers. 

% getting required code. 
addpath(genpath('~/'))

% getting subjects. 
task = 'prob_sel2';
subjects = get_subs(task);

% alignment parameters
durationMS = 4000;
offsetMS = -1000;
bufferMS = 0; % seems like this should stay = 0
notchFreqs = [55 65];
filtType = 'stop';
filtOrder = 4;
resampleFo = 500;

% time vector. 
tmsec=offsetMS:(1000/resampleFo):offsetMS+durationMS-1;

%% time periods for sliding window decode. (all in milliseconds) 
% [20160304] starting with a large window and then will scale down to test sensitivity.
windowSize = 500;
stepSize = 10;
timePeriods = [0:stepSize:durationMS;(0:stepSize:durationMS)+windowSize];
timePeriods = timePeriods(:,timePeriods(2,:)<durationMS);

% looping over subjects. The ECoG subjects start at 67 for ProbSel2, then there are about 40+ patients before another block of empties. The first 66 patients are single unit recordings.  
for sb = 67:length(subjects)
	
	% getting subject events. 
	[events evDeets] = get_sub_events(task,subjects{sb}); 

	% [20160301] getting event types. 
	evType = getStructField(events,'evType');
	stimIdx = cellfun(@isequal,evType,repmat({'stim'},size(evType)));
        fbIdx = cellfun(@isequal,evType,repmat({'fb'},size(evType)));

	% [20160301] getting specific event times. 
	stimTimes = [events(stimIdx).mstime];
	fbTimes = [events(fbIdx).mstime];  	
		
	% finding all pairs of electrodes...
	trodePairs = nchoosek(evDeets.leads,2);
	% looping over channels
    for ch = evDeets.leads	
		try % to get EEG. 
			% [20160301] get full bandwidth ECoG data. 
			[EEG resampleFs] = gete_ms(ch,events,durationMS,offsetMS,bufferMS,notchFreqs,filtType,filtOrder,resampleFo);
		
			% % filter in the high gamma range. 	
        	% [highGamma resampleFs_hg] = gete_ms(ch,events,durationMS,offsetMS,bufferMS,[70 150],'bandpass');
			
        	% tensorizing data 
			if isequal(size(EEG,2),0)
				display(sprintf('EEG is empty for subject %s.',events(1).subject))
			else
				datamat(:,:,ch) = EEG';
			end

		catch % otherwise let me know it's not available. 
			display(sprintf('no file for electrode %d',ch))
		end
	end % looping over channels

	%% for each channel pair, calculate CMI
	parfor prs = 1:size(trodePairs,1)
		%% calulating sCMI for each time period
        for tper = 1:size(timePeriods,2)
   			timePeriod = timePeriods(:,tper)';
    		display(['Calculating information transfer between ' num2str(timePeriod(1)) ' and ' num2str(timePeriod(2)) 'seconds for Probabilistic Selection Task.']);
    		timePeriod = int16((timePeriod)+1); % convert to sample indices
    		
    		% these are the data for sCMI calculation. They are shuffled below. 
    		Data1 = squeeze(mean(datamat(timePeriod(1):timePeriod(2),fbIdx,trodePairs(prs,1))),3));
    		Data2 = squeeze(mean(datamat(timePeriod(1):timePeriod(2),fbIdx,trodePairs(prs,2))),3));
    
    		% randomly shuffling the samples of the data in [Data1] and [Data2]
    		% not shuffling trials in order to keep the feedback structure the same. It remains to be seen whether this will affect CMI. 
    		Ti = find(fbIdx);
    		for sh = 1:length(Ti)
    			timeBasis = timePeriod(1):timePeriod(2);
    			randIdx = randperm(length(timeBasis));
    			shuffData1(:,sh) = datamat(timeBasis(randIdx),Ti(sh),trodePairs(prs,1))));
    			randIdx = randperm(length(timeBasis));
    			shuffData2(:,sh) = datamat(timeBasis(randIdx),Ti(sh),trodePairs(prs,2))));
    		end
    
    		%% [20160304] actually doing CMI.
			[CMI_FB(:,tper,prs,sb) MI_FB(tper,prs,sb)] = scmi(Data1,Data2,fbClass)
			
			%% [20160304] running the same code on suffled data. 
    		[shuffCMI_FB(:,tper,prs,pts,sb) shuffMI_FB(tper,prs,sb)] = sCMI(shuffData1, shuffData2, fbClass);

    	end
	end % looping over pairwise combos of trodes

	%% save data for visualization with other scripts...
	saveStr = sprintf('CMIoverSubjectsforProbabilisticSelectionTask.mat')
	save(saveStr,'CMI_FB','MI_FB','shuffCMI_FB','shuffMI_FB','trodePairs','stepSize','windowSize','durationMS','offsetMS','resampleFo','resampleFs')

	clear datamat EEG 

end % looping over subjects. 



