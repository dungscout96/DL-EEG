% Add EEGLAB path
addpath('/expanse/projects/nemar/eeglab');
eeglab; close;

try, parpool; end

% Path to data
folderin  = '/expanse/projects/nemar/child-mind-uncompressed';
folderout = '/expanse/projects/nemar/child-mind-restingstate-preprocessed';
folders = dir(folderin);
fprintf('length %f', length(folders));

% read the CSV file
disp('Loading info file...');
info = loadtxt('HBN_all_Pheno.csv', 'delim', ',', 'verbose', 'off');
info = info(2:end,:);

issueFlag = cell(1, length(folders));
N = length(folders);
subj_data   = cell(1,N);
subj_vval = cell(1,N);

for iFold = 1:N
    fileName = fullfile(folders(iFold).folder, folders(iFold).name, 'EEG/raw/mat_format/RestingState.mat');
    fileNameClosedSet = fullfile(folderout, [ folders(iFold).name '_eyesclosed.set' ]);
    fileNameOpenSet   = fullfile(folderout, [ folders(iFold).name '_eyesopen.set' ]);
    
    %if exist(fileNameClosedSet, 'file')
	%disp([fileNameClosedSet ' exists. Skipped']);
	%continue;
        %delete(fileNameClosedSet);
        %delete(fileNameClosedFdt);
    %end
    %if exist(fileNameOpenSet, 'file')
	%disp([fileNameClosedSet ' exists. Skipped']);
	%continue;
        %delete(fileNameOpenSet);
        %delete(fileNameOpenFdt);
    %end
    infoRow = strmatch(folders(iFold).name, info(:,1)', 'exact');
    if exist(fileName, 'file') && length(infoRow) > 0
        try
            EEG = load(fileName);
            EEG = EEG.EEG;
            
            if EEG.nbchan == 129
                for iEvent2 = 1:length(EEG.event)
                    EEG.event(iEvent2).latency = EEG.event(iEvent2).sample;
                end
                
                % copy info
                EEG.gender     = info{infoRow(1),2};
                EEG.age        = info{infoRow(1),3};
                EEG.handedness = info{infoRow(1),4};
                EEG.subjID     = folders(iFold).name;
                % get channel location and clean data
                % chanlocs = pop_chanedit(struct('labels', {'cz' 'pz' 'fz' 'nz' 'tp9' 'tp10' 'lpa' 'rpa'})) 
                % mastoids in eeg system is TP9 (maybe 68/69) TP10 
                % then go on web: electrical geodesics 128 channel
                % pop_chancoresp(chanlocs, EEG.chanlocs)
                % plugin to do EOG-regression based 
                EEG = pop_chanedit(EEG, 'load',{'GSN_HydroCel_129.sfp','filetype','autodetect'});
                EEG = pop_chanedit(EEG, 'setref',{'1:129','Cz'});
                EEG = pop_rmbase(EEG, []);
                EEG = pop_eegfiltnew(EEG, 'locutoff',0.01,'hicutoff',64); % Wouldn't this leave linear trend and 
                EEG = pop_resample(EEG, 125);
                EEG = eeg_checkset( EEG );
                EEG = pop_reref( EEG, [57 100], 'keepref', 'on' );
                EEG2 = pop_clean_rawdata(EEG, 'FlatlineCriterion',5,'ChannelCriterion',0.7,'LineNoiseCriterion',4,'Highpass','off','BurstCriterion','off','WindowCriterion','off','BurstRejection','off','Distance','Euclidian');
                EEG = pop_interp(EEG2, EEG.chanlocs);
                
%                 % resting state: instruction. Open 20 close 40 20 - 40
%                 % wait 2.5 seconds. 
                EEGeyeso = pop_epoch( EEG, {  '20  ' }, [3  17], 'newname', 'Eyes open', 'epochinfo', 'yes');
                EEGeyeso.eyesclosed = 0;
                EEGeyeso = eeg_epoch2continuous(EEGeyeso);
                EEGeyeso.event = []; % removing boundary events
                EEGeyeso = eeg_regepochs( EEGeyeso, 'recurrence', 2, 'limits', [0 2]);
                if EEGeyeso.trials ~= 35
                    error('Missing eyes-opened segment')
                end
                pop_saveset(EEGeyeso, 'filename', fileNameOpenSet, 'savemode', 'onefile');
                
                EEGeyesc = pop_epoch( EEG, {  '30  ' }, [3  37], 'newname', 'Eyes closed', 'epochinfo', 'yes');
                EEGeyesc.eyesclosed = 1;
                EEGeyesc = eeg_epoch2continuous(EEGeyesc);
                EEGeyesc.event = []; % removing boundary events
                EEGeyesc = eeg_regepochs( EEGeyesc, 'recurrence', 2, 'limits', [0 2]);
                if EEGeyesc.trials ~= 85
                    error('Missing eyes-closed segment')
                end
                pop_saveset(EEGeyesc, 'filename', fileNameClosedSet, 'savemode', 'onefile');
                
                % add to data matrix
                subj_data{iFold} = EEGeyesc.data;
                subj_vval{iFold} = repelem({EEG.subjID}, size(EEGeyesc.data,3));
            else
                issueFlag{iFold} = 'Not 129 channels';
            end
        catch
            issueFlag{iFold} = lasterr;
        end
    end
end

indIssue = find(~cellfun(@isempty, issueFlag));
fprintf('Issues at indices (empty means no issues): %s\n', int2str(indIssue));
if ~isempty(indIssue), issueFlag(indIssue)', end

% save matrix
X = cat(3,subj_data{:});
Y = cat(2,subj_vval(:));

param_text = '_2s';
param_text = [param_text '_128chan'];
param_text = [param_text '_raw'];

save(['child_mind_data' param_text '.mat'],'X','-v7.3');
save(['child_mind_subjIDs' param_text '.mat'],'Y','-v7.3');
clear % clearing var is always good
