function load_data_master(winLength, numChan, isSpectral, isTopo)
%clear
isTesting = true;
if ~isTesting
    % Add EEGLAB path
    addpath('/expanse/projects/nemar/eeglab');
    %addpath('./eeglab');
    eeglab; close;

    try, parpool(23); end
end

if isTesting
    folderout = '.';
else
    folderout = '/expanse/projects/nemar/child-mind-restingstate-preprocessed';
end

fileNamesClosed = dir(fullfile(folderout, '*_eyesclosed.set'));
female = readtable('female.csv');
female = female.Var1;
male = readtable('male.csv');
male = male.Var1;
if isTesting
    N = 7;
else
    N = length(female)*2;
end
disp(N);
% choose training, validation, and test from different subjects.
N_test_subjs = ceil(N * 0.125);
N_val_subjs = ceil(N * 0.3125);
N_train_subjs = N - N_test_subjs - N_val_subjs;

subj_data = cell(1,N);
subj_gender = cell(1,N);
subj_IDs = cell(1,N);

% dimension of number of sample in the data. If topo map, 4 (rgb x samples), otherwise 3
% (chan x times x samples)
if isTopo, sample_dim = 4; else, sample_dim = 3; end

for iSubj=1:N 
    if isTesting
        EEGeyesc = pop_loadset('filepath', folderout, 'filename', fileNamesClosed(iSubj).name);
    else
        if mod(iSubj,2) == 1
            % female
            EEGeyesc = pop_loadset('filepath', folderout, 'filename', [female{floor(iSubj/2)+1} '_eyesclosed.set']);
        else
            % male
            EEGeyesc = pop_loadset('filepath', folderout, 'filename', [male{iSubj/2} '_eyesclosed.set']);
        end
    end
    if ~strcmp(EEGeyesc.filename,'NDAREE675XRY_eyesclosed.set') &&~strcmp(EEGeyesc.filename,'NDARFA860RPD_eyesclosed.set') &&~strcmp(EEGeyesc.filename,'NDARMR277TT7_eyesclosed.set')&&~strcmp(EEGeyesc.filename,'NDARMP784KKE_eyesclosed.set')&&~strcmp(EEGeyesc.filename,'NDARNK241ZXA_eyesclosed.set')
    % sub-sample using window length
    EEGeyesc = eeg_regepochs( EEGeyesc, 'recurrence', winLength, 'limits', [0 winLength]);
    tmpdata = EEGeyesc.data;
    chanlocs = EEGeyesc.chanlocs;

    % If numChan is 24, sub-select channel. Otherwise assuming it's 128
    % which is the original data
    if numChan == 24
        % sub-select channel
        channel_map = {'Fp1', 22; 'Fp2', 9; 'F7', 33;'F3',24;'Fz', 11;'F4',124;'F8', 122;'FC3', 29;'FCz', 6;'FC4', 111;'T3', 45;'C3', 36;
            'C4', 104;'T4', 108;'CP3', 42;'CPz', 55;'CP4', 93;'T5', 58;'P3', 52;'Pz', 62;'P4', 92;'T6', 96;'O1', 70; 'Cz', 'Cz'};
        chanindices = [];
        for iChannel = 1:size(channel_map,1)
            if ~ischar(channel_map{iChannel,2})
                egiChannel = sprintf('E%d', channel_map{iChannel,2});
                chanindices = [chanindices find(cellfun(@(x) strcmp(x,egiChannel), {EEGeyesc.chanlocs.labels}))];
            else
                chanindices = [chanindices find(cellfun(@(x) strcmp(x,'Cz'), {EEGeyesc.chanlocs.labels}))];
            end
        end
        if (length(chanindices) < 24)
        warning('%s have missing channels. Skipped', fileNamesClosed(iFile).name);
            disp(size(tmpdata));
        continue;
        end 
        tmpdata = tmpdata(chanindices,:,:);
	chanlocs = EEGeyesc.chanlocs(chanindices);
    end
    disp(size(tmpdata));

    % If compute spectral
    if isSpectral
        finalData = zeros(size(tmpdata));
        for epoch=1:size(tmpdata,3)
            nSamples = winLength*EEGeyesc.srate;
            data = tmpdata(1:numChan,1:nSamples);
            taperedData = bsxfun(@times, data', hamming(nSamples));
            fftData = fft(taperedData);
            stopIdx = nSamples/2;
            fftData(stopIdx+1:end,:) = [];
            logPowerData = log(abs(fftData').^2);
            logPowerZeroedData = bsxfun(@minus, logPowerData, mean(logPowerData')');
            phaseData    = angle(fftData');
            finalData(:,:,epoch) = [ logPowerZeroedData phaseData];
        end
        tmpdata = finalData;
    elseif isTopo
        tmp_topo = cell(1,size(tmpdata,3));
        parfor s=1:size(tmpdata,3)
            freqRanges = [4 7; 7 13; 14 30]; % frequencies, but also indices
            % compute spectrum
            srates = 128;
            [XSpecTmp,~] = spectopo(tmpdata(:,:,s), winLength*srates, srates, 'plot', 'off', 'overlap', 50);
            XSpecTmp(:,1) = []; % remove frequency 0

            % get frequency bands
            theta = mean(XSpecTmp(:, freqRanges(1,1):freqRanges(1,2)), 2);
            alpha = mean(XSpecTmp(:, freqRanges(2,1):freqRanges(2,2)), 2);
            beta  = mean(XSpecTmp(:, freqRanges(3,1):freqRanges(3,2)), 2);

            % get grids
            [~, gridTheta] = topoplot( theta, chanlocs, 'verbose', 'off', 'gridscale', 24, 'noplot', 'on', 'chaninfo', EEGeyesc(1).chaninfo);
            [~, gridAlpha] = topoplot( alpha, chanlocs, 'verbose', 'off', 'gridscale', 24, 'noplot', 'on', 'chaninfo', EEGeyesc(1).chaninfo);
            [~, gridBeta ] = topoplot( beta , chanlocs, 'verbose', 'off', 'gridscale', 24, 'noplot', 'on', 'chaninfo', EEGeyesc(1).chaninfo);
            gridTheta = gridTheta(end:-1:1,:); % for proper imaging using figure; imagesc(grid);
            gridAlpha = gridAlpha(end:-1:1,:); % for proper imaging using figure; imagesc(grid);
            gridBeta  = gridBeta( end:-1:1,:); % for proper imaging using figure; imagesc(grid);

            topoTmp = gridTheta;
            topoTmp(:,:,3) = gridBeta;
            topoTmp(:,:,2) = gridAlpha;
            topoTmp = single(topoTmp);

            % remove Nan
            minval = nanmin(nanmin(topoTmp,[],1),[],2);
            maxval = nanmax(nanmax(topoTmp,[],1),[],2);

            % transform to RGB image
            topoTmp = bsxfun(@rdivide, bsxfun(@minus, topoTmp, minval), maxval-minval)*255;
            topoTmp(isnan(topoTmp(:))) = 0;
            tmp_topo{s} = topoTmp;
        end
        tmpdata = cat(4,tmp_topo{:});
>>>>>>> 7fadf8a5d3fc3a32151bc5d96fd3e911a7df7a30
    end

    % append to XOri
    subj_data{iSubj} = tmpdata;
    subj_gender{iSubj} = repelem(EEGeyesc.gender, size(tmpdata,sample_dim));
    subj_IDs{iSubj} = repelem(string(EEGeyesc.subjID), size(tmpdata,sample_dim));
    end
end

% split into train, val, test
X_test = cat(sample_dim,subj_data{1:N_test_subjs});
disp(size(X_test));
Y_test = cat(2,subj_gender{1:N_test_subjs});
disp(size(Y_test));
test_subjID = cat(2,subj_IDs{1:N_test_subjs});
X_val = cat(sample_dim,subj_data{N_test_subjs+1:N_test_subjs + N_val_subjs});
Y_val = cat(2,subj_gender{N_test_subjs+1:N_test_subjs + N_val_subjs});
X_train = cat(sample_dim,subj_data{N_test_subjs + N_val_subjs + 1:end});
Y_train = cat(2,subj_gender{N_test_subjs + N_val_subjs + 1:end});

% save
param_text = ['_' num2str(winLength) 's'];
param_text = [param_text '_' num2str(numChan) 'chan'];
if isSpectral
    param_text = [param_text '_spectral'];
elseif isTopo
    param_text = [param_text '_topo'];
else
    param_text = [param_text '_raw'];
end
save(['child_mind_x_train' param_text '.mat'],'X_train','-v7.3');
save(['child_mind_y_train' param_text '.mat'],'Y_train','-v7.3');
save(['child_mind_x_val' param_text '.mat'],'X_val','-v7.3');
save(['child_mind_y_val' param_text '.mat'],'Y_val','-v7.3');
save(['child_mind_x_test' param_text '.mat'],'X_test','-v7.3');
save(['child_mind_y_test' param_text '.mat'],'Y_test','-v7.3');
save('test_subj.mat','test_subjID','-v7.3');

