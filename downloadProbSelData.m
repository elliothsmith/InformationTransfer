%% This code implements the UPenn & iEEG.org functions in order to run sCMI (or other kind of CMI?) on the data for the probilisitic reinforcement learning task as in Frank, Ramayya, and Zaghloul papers. 

clear all
clc


% getting required code. 
addpath(genpath('~/'))

% getting subjects. 
task = 'prob_sel2';
subjects = get_subs(task);

% alignment parameters
durationMS = 6000;
offsetMS = -2000;
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

	% getting feeback classes. 
	fbClass = getStructField(events,'correct');
	for jz = 1:length(fbClass)
		if isstr( fbClass{jz})
		tmp(jz) = 0;
		else
		tmp(jz) = fbClass{jz};
		end
	end
	fbClass = tmp;

	% [20160301] getting specific event times. 
	stimTimes = [events(stimIdx).mstime];
	fbTimes = [events(fbIdx).mstime];
	
if ~isempty(evDeets.leads)
	for ch = evDeets.leads
		try
			Data(:,:,ch) = gete_ms(ch,events,durationMS,offsetMS,0,notchFreqs,filtType,filtOrder,resampleFo);
		catch
			display(sprintf('gete_ms failed for subject %s',subjects{sb}))
		end
	end % looping over channels

	%% save data for visualization with other scripts...
	saveStr = sprintf('~/Data/ProbSel/tensorizedProbabilisticSelectionTaskData_%s.mat',subjects{sb})
	save(saveStr,'Data','resampleFo','events','evDeets','tmsec','evType','stimTimes','fbTimes','stimIdx','fbIdx','-v7.3')	
	
	clear Data events fID
else
	display(sprintf('no leads found for patient %s',subjects{sb}));
end

end % looping over subjects. 


