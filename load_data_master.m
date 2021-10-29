% Functions should be re-entrant (use no hard coded info). This code is better as a script.

opt.infofile = 'HBN_NEMAR_Pheno.csv';
opt.folder   = '/expanse/projects/nemar/child-mind-restingstate-preprocessed';

if ~exist('opt', 'var'), opt = []; end
if ~isfield(opt, 'winlength'),  opt.winlength = 2; end
if ~isfield(opt, 'numchan'),    opt.numchan    = 24; end
if ~isfield(opt, 'isspectral'), opt.isspectral = false; end
if ~isfield(opt, 'istopo'),     opt.istopo     = false; end
if ~isfield(opt, 'folder'),     opt.folder      = '.'; end
if ~isfield(opt, 'variable'),   opt.variable   = 'Sex'; end
if ~isfield(opt, 'infofile'),   opt.infofile   = 'HBN_all_Pheno.csv'; end % make a test file with a few subjects for testing

if ~exist('pop_loadset.m', 'file')
    warning('You must add EEGLAB to your path before calling this script');
    fprintf('Using EEGLAB on Expanse...');
    addpath('/expanse/projects/nemar/eeglab');
    eeglab; close;
    try, parpool; end
end

info = readtable(opt.infofile);
varValues = info.(opt.variable);
if length(unique(varValues)) ~= 2
    error('Can only process binary variables');
end

var1 = info.EID(varValues == 0);
var2 = info.EID(varValues == 1);
if length(var1) > length(var2)
    N = length(var2)*2;
else
    N = length(var1)*2;
end

% choose training, validation, and test from different subjects.
N_test_subjs  = ceil(N * 0.125);
N_val_subjs   = ceil(N * 0.3125);
N_train_subjs = N - N_test_subjs - N_val_subjs;

subj_data   = cell(1,N);
subj_vval = cell(1,N);
subj_ages   = cell(1,N);
subj_IDs    = cell(1,N);

% dimension of number of sample in the data. If topo map, 4 (rgb x samples), otherwise 3
% (chan x times x samples)
if opt.istopo, sample_dim = 4; else, sample_dim = 3; end

parfor iSubj=1:N
    varVal = mod(iSubj,2);
    if varVal == 1
        % var2 (male=1 when Sex is choosen)
        EEGeyesc = pop_loadset('filepath', opt.folder, 'filename', [var2{floor(iSubj/2)+1} '_eyesclosed.set']);
    else
        % var1 (female=0)
        EEGeyesc = pop_loadset('filepath', opt.folder, 'filename', [var1{iSubj/2} '_eyesclosed.set']);
    end
    if ~strcmp(EEGeyesc.filename,'NDAREE675XRY_eyesclosed.set') &&~strcmp(EEGeyesc.filename,'NDARFA860RPD_eyesclosed.set') &&~strcmp(EEGeyesc.filename,'NDARMR277TT7_eyesclosed.set')&&~strcmp(EEGeyesc.filename,'NDARMP784KKE_eyesclosed.set')&&~strcmp(EEGeyesc.filename,'NDARNK241ZXA_eyesclosed.set') && ~strcmp(EEGeyesc.filename,'NDARFB322DRA_eyesclosed.set')
        % sub-sample using window length
        EEGeyesc = eeg_epoch2continuous(EEGeyesc);
        EEGeyesc = eeg_regepochs( EEGeyesc, 'recurrence', opt.winlength, 'limits', [0 opt.winlength]);
        tmpdata = EEGeyesc.data;
        chanlocs = EEGeyesc.chanlocs;
        
        % If opt.numchan is 24, sub-select channel. Otherwise assuming it's 128
        % which is the original data
        if opt.numchan == 24
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
        if opt.isspectral
            finalData = zeros(size(tmpdata));
            for epoch=1:size(tmpdata,3)
                nSamples = opt.winlength*EEGeyesc.srate;
                data = tmpdata(1:opt.numchan,1:nSamples);
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
        elseif opt.istopo
            tmp_topo = cell(1,size(tmpdata,3));
            disp(size(tmpdata,3));
            for s=1:size(tmpdata,3)
                freqRanges = [4 7; 7 13; 14 25]; % frequencies, but also indices
                % compute spectrum
                srates = 128;
                [XSpecTmp,~] = spectopo(tmpdata(:,:,s), opt.winlength*srates, srates, 'plot', 'off', 'overlap', 50);
                XSpecTmp(:,1) = []; % remove frequency 0
                
                % get frequency bands
                theta = mean(XSpecTmp(:, freqRanges(1,1):freqRanges(1,2)), 2);
                alpha = mean(XSpecTmp(:, freqRanges(2,1):freqRanges(2,2)), 2);
                beta  = mean(XSpecTmp(:, freqRanges(3,1):freqRanges(3,2)), 2);
%                 disp('theta chan');
%                 size(theta)
%                 disp('alpha chan');
%                 size(alpha)
%                 disp('beta chan');
%                 size(beta)
                if sum(theta,'all') == 0
                    error('0 theta');
                end
                if sum(alpha,'all') == 0
                    error('0 alpha');
                end
                if sum(beta,'all') == 0
                    error('0 beta');
                end
                
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
        end
        
        % append to XOri
        subj_data{iSubj} = tmpdata;
        subj_vval{iSubj} = repelem(varVal, size(tmpdata,sample_dim));
    end
end

% split into train, val, test
X_test  = cat(sample_dim,subj_data{1:N_test_subjs});
Y_test  = cat(         2,subj_vval{1:N_test_subjs});
fprintf('\nTesting size X -> %s\n', sprintf('%d ', size(X_test)) );
fprintf(  'Testing size Y -> %s\n', sprintf('%d ', size(Y_test)) );
X_val   = cat(sample_dim,subj_data{N_test_subjs+1:N_test_subjs + N_val_subjs});
Y_val   = cat(         2,subj_vval{N_test_subjs+1:N_test_subjs + N_val_subjs});
X_train = cat(sample_dim,subj_data{N_test_subjs + N_val_subjs + 1:end});
Y_train = cat(         2,subj_vval{N_test_subjs + N_val_subjs + 1:end});

% save
param_text = ['_' num2str(opt.winlength) 's'];
param_text = [param_text '_' num2str(opt.numchan) 'chan'];
if opt.isspectral
    param_text = [param_text '_spectral'];
elseif opt.istopo
    param_text = [param_text '_topo'];
else
    param_text = [param_text '_raw'];
end
save(['child_mind_x_train' param_text '.mat'],'X_train','-v7.3');
save(['child_mind_y_train' param_text '.mat'],'Y_train','-v7.3');
save(['child_mind_x_val'   param_text '.mat'],'X_val'  ,'-v7.3');
save(['child_mind_y_val'   param_text '.mat'],'Y_val'  ,'-v7.3');
save(['child_mind_x_test'  param_text '.mat'],'X_test' ,'-v7.3');
save(['child_mind_y_test'  param_text '.mat'],'Y_test' ,'-v7.3');
clear % clearing var is always good
