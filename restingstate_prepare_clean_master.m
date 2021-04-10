% Add EEGLAB path
addpath('/expanse/projects/nemar/eeglab');
eeglab; close;

try, parpool; end

% Path to data
folderin  = '/expanse/projects/nemar/child-mind-uncompressed';
folderout = '/expanse/projects/nemar/child-mind-restingstate-preprocessed';
folders = dir(folderin);
fprintf('length %f', length(folders));

% Prepare data
fileTrain = dir(fullfile(folderout, '*.set'));
% read the CSV file
disp('Loading info file...');
info = loadtxt('HBN_all_Pheno.csv', 'delim', ',', 'verbose', 'off');
info = info(2:end,:);

XTrain = cell(1,length(folders));
YTrain = cell(1,length(folders));
issueFlag = cell(1, length(folders));
count = 1;
parfor iFold = 1:length(folders)
    %for iFold = 1:6 %length(folders)
    
    fileName = fullfile(folders(iFold).folder, folders(iFold).name, 'EEG/raw/mat_format/RestingState.mat');
    fileNameClosedSet = fullfile(folderout, [ folders(iFold).name '_eyesclosed.set' ]);
    fileNameClosedFdt = fullfile(folderout, [ folders(iFold).name '_eyesclosed.fdt' ]);
%     fileNameOpenSet   = fullfile(folderout, [ folders(iFold).name '_eyesopen.set' ]);
%     fileNameOpenFdt   = fullfile(folderout, [ folders(iFold).name '_eyesopen.fdt' ]);
    if exist(fileNameClosedSet, 'file')
	disp([fileNameClosedSet ' exists. Skipped']);
	continue;
        %delete(fileNameClosedSet);
        %delete(fileNameClosedFdt);
    end
    %if exist(fileNameOpenSet, 'file')
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
                EEG = pop_select(EEG, 'nochannel', 129);
                EEG = pop_rmbase(EEG, []);
                EEG = pop_eegfiltnew(EEG, 'locutoff',0.25,'hicutoff',25);
                EEG = pop_resample(EEG, 128);
                % add Cz back and re-reference
                EEG = pop_chanedit(EEG, 'append',131,'changefield',{132,'labels','Cz'},'changefield',{132,'theta','0'},'changefield',{132,'radius','0'},'changefield',{132,'X','5.2047e-15'},'changefield',{132,'Y','0'},'changefield',{132,'Z','85'},'changefield',{132,'sph_theta','0'},'changefield',{132,'sph_phi','90'},'changefield',{132,'sph_radius','85'},'setref',{'1:131','Cz'});
                EEG = eeg_checkset( EEG );EEG = pop_reref( EEG, [],'refloc',struct('labels',{'Cz'},'Y',{0},'X',{5.2047e-15},'Z',{85},'sph_theta',{0},'sph_phi',{90},'sph_radius',{85},'theta',{0},'radius',{0},'type',{''},'ref',{''},'urchan',{[]},'datachan',{0}));
                EEG = pop_reref( EEG, [57 100] );

                
%                 % resting state: instruction. Open 20 close 40 20 - 40
%                 % wait 2.5 seconds. 
%                 EEGeyeso = pop_epoch( NEWEEG, {  '20  ' }, [4  19], 'newname', 'Eyes open', 'epochinfo', 'yes');
%                 EEGeyeso.eyesclosed = 0;
%                 pop_saveset(EEGeyeso, fileNameOpenSet);
                EEGeyesc = pop_epoch( EEG, {  '30  ' }, [3  37], 'newname', 'Eyes closed', 'epochinfo', 'yes');
                EEGeyesc.eyesclosed = 1;
                EEGeyesc = eeg_epoch2continuous(EEGeyesc);
                EEGeyesc2 = pop_clean_rawdata(EEGeyesc, 'FlatlineCriterion',5,'ChannelCriterion',0.7,'LineNoiseCriterion',4,'Highpass','off','BurstCriterion','off','WindowCriterion','off','BurstRejection','off','Distance','Euclidian');
                EEGeyesc2 = pop_interp(EEGeyesc2, EEGeyesc.chanlocs);
                pop_saveset(EEGeyesc2, fileNameClosedSet);
  
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
