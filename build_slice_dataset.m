%% ==========================================
% BUILD_SLICE_DATASET
% Build a 2D slice dataset from BraTS cases
%
% Output:
%   - images: H x W x 1 x N single
%   - masks:  H x W x 1 x N logical
%   - meta:   table with case/slice metadata
%
% Assumptions:
%   - each patient folder contains NIfTI files
%   - load_brats_case.m is available on MATLAB path
%
% Recommended first use:
%   - modality = 'flair'
%   - target = whole tumor (seg > 0)
% ==========================================

clear; clc; close all;

%% -------------------------------
% 1. User settings
% -------------------------------

dataRoot = 'data';                 % folder containing all patient folders
outputFile = 'brats2d_dataset.mat';

modality = 'flair';                % 'flair', 't1', 't1ce', 't2'
resizeTo = [256 256];              % output slice size, DL models need consistent input size, easier batching and training
minBrainPixels = 500;              % skip a slice that has fewer than 500 nonzero brain pixels
minTumorPixels = 20;               % tumor threshold for "tumor slice"
keepMode = 'balanced';             % 'all', 'tumor_only', 'balanced'
maxNonTumorPerCase = 20;           % used only when keepMode='balanced', otherwise the model may see too many negative slices and learn to predict empty masks

verbose = true;

%% -------------------------------
% 2. Discover case folders
% -------------------------------
entries = dir(dataRoot);
isCaseDir = [entries.isdir] & ~ismember({entries.name}, {'.', '..'});

caseDirs = entries(isCaseDir);
numCases = numel(caseDirs);

if numCases == 0
    error('No case folders found under "%s".', dataRoot);
end

fprintf('Found %d case folders.\n', numCases);

%% -------------------------------
% 3. Storage containers
% -------------------------------
imageList = {};        % cell array for storing slices
maskList = {};         % cell array for storing corresponding masks

caseIDList = {};
sliceIdxList = [];
hasTumorList = [];
brainPixelCountList = [];
tumorPixelCountList = [];

%% -------------------------------
% 4. Loop over cases
% -------------------------------
for c = 1:numCases
    caseName = caseDirs(c).name;
    caseFolder = fullfile(caseDirs(c).folder, caseDirs(c).name);

    fprintf('\n[%d/%d] Processing case: %s\n', c, numCases, caseName);

    try
        data = load_brats_case(caseFolder);
    catch ME
        warning('Skipping case "%s" due to load error:\n%s', caseName, ME.message);
        continue;
    end         %try/catch: if one patient folder is broken, the script does not crash

    % --- Select modality ---
    if ~isfield(data, modality)
        warning('Case "%s" missing modality "%s". Skipping.', caseName, modality);
        continue;
    end

    vol = data.(modality);

    if ~isfield(data, 'seg')
        warning('Case "%s" missing segmentation. Skipping.', caseName);
        continue;
    end          % masks needed for supervised learning, so skip unlabeled patients

    seg = data.seg;
    wholeTumor = seg > 0;     % converts BraTS labels 1, 2, and 4 into one binary whole-tumor mask.

    % --- Sanity check sizes ---
    if ~isequal(size(vol), size(seg))
        warning('Size mismatch in case "%s". Skipping.', caseName);
        continue;
    end               % Checks that MRI and segmentation mask have exactly the same size.

    % --- Normalize MRI volume using nonzero voxels only ---
    volNorm = normalize_mri_volume(vol);

    % --- Determine valid slices ---
    numSlices = size(volNorm, 3);

    tumorSliceIndices = [];
    nonTumorSliceIndices = [];

    % inspecting every axial slice and each valid slice goes into either
    % tumor slice list or non-tumor slice list
    for z = 1:numSlices
        imgSlice = volNorm(:, :, z);
        maskSlice = wholeTumor(:, :, z);      %extract one 2D image and 2D mask

        brainPixels = nnz(vol(:, :, z) > 0);   % use original volume for nonzero brain support
        tumorPixels = nnz(maskSlice);

        % Counts how many nonzero pixels exist in that slice, if very few,
        % the slice is nearly empty
        if brainPixels < minBrainPixels
            continue;
        end

        % Counts how many tumor pixels are in that slice and skip mostly
        % empty slices
        if tumorPixels >= minTumorPixels
            tumorSliceIndices(end+1) = z; %#ok<SAGROW>
        else
            nonTumorSliceIndices(end+1) = z; %#ok<SAGROW>
        end
    end

    % --- Select slices based on keepMode ---
    switch lower(keepMode)
        case 'all'
            selectedSlices = sort([tumorSliceIndices, nonTumorSliceIndices]);

        case 'tumor_only'
            selectedSlices = tumorSliceIndices;


        % keeps all tumor slices and a limited number of non-tumoro slices
        % this reduces class imbalance and helps the model not just learn
        % background
        case 'balanced'
            nTumor = numel(tumorSliceIndices);
            nNonTumorAvailable = numel(nonTumorSliceIndices);

            if nTumor == 0
                % keep a small sample of non-tumor slices if there is no tumor in valid slices
                nTake = min(maxNonTumorPerCase, nNonTumorAvailable);
                selectedNonTumor = nonTumorSliceIndices(1:nTake);
            else
                nTake = min([nTumor, maxNonTumorPerCase, nNonTumorAvailable]);
                selectedNonTumor = nonTumorSliceIndices(1:nTake);
            end

            % If there are no tumor slices, keeps a small sample of
            % non-tumor slices, this gives the final slices for that
            % patients
            selectedSlices = sort([tumorSliceIndices, selectedNonTumor]);

        otherwise
            error('Unknown keepMode: %s', keepMode);
    end

    if isempty(selectedSlices)
        warning('No slices selected for case "%s".', caseName);
        continue;
    end

    % --- Extract and store slices ---
    for z = selectedSlices
        imgSlice = volNorm(:, :, z);
        maskSlice = wholeTumor(:, :, z);

        brainPixels = nnz(vol(:, :, z) > 0);
        tumorPixels = nnz(maskSlice);
        hasTumor = tumorPixels >= minTumorPixels;

        % Resize
        imgResized = imresize(imgSlice, resizeTo, 'bilinear');     % for image, bilinear interpolation is appropriate because MRI intensities are continuous
        maskResized = imresize(maskSlice, resizeTo, 'nearest');    % for mask, nearest is required because masks are labels

        % Ensure correct types
        imgResized = single(imgResized);     % single is efficient for DL
        maskResized = logical(maskResized);  % logical is efficient for binary masks

        % Store
        imageList{end+1,1} = imgResized; %#ok<SAGROW>
        maskList{end+1,1}  = maskResized; %#ok<SAGROW>

        %metadata
        caseIDList{end+1,1} = caseName; %#ok<SAGROW>
        sliceIdxList(end+1,1) = z; %#ok<SAGROW>
        hasTumorList(end+1,1) = hasTumor; %#ok<SAGROW>
        brainPixelCountList(end+1,1) = brainPixels; %#ok<SAGROW>
        tumorPixelCountList(end+1,1) = tumorPixels; %#ok<SAGROW>
    end

    if verbose
        fprintf('  Tumor slices found: %d\n', numel(tumorSliceIndices));
        fprintf('  Non-tumor valid slices found: %d\n', numel(nonTumorSliceIndices));
        fprintf('  Selected slices saved: %d\n', numel(selectedSlices));
    end
end

%% -------------------------------
% 5. Convert cell arrays to 4D arrays
% -------------------------------
numSamples = numel(imageList);     % Number of total slides collected

if numSamples == 0
    error('No slices were collected. Check your paths and thresholds.');
end

H = resizeTo(1);
W = resizeTo(2);

% the singlrton third dimension is used because MATLAB DL functions expect
% channel dimension; for grayscale images, channels=1
images = zeros(H, W, 1, numSamples, 'single');     % arrays in the format expected by training code: H X W X 1 X N; #rowsorimageheight*#columnsorimagewidth*#modalities*#slices (DL expects HxWxChannelsxBatch); In our case single-channel FLAIR MRI slices
masks  = false(H, W, 1, numSamples);

% Copying from cell arrays to tensors
for i = 1:numSamples
    images(:, :, 1, i) = imageList{i};
    masks(:, :, 1, i)  = maskList{i};
end

%% -------------------------------
% 6. Metadata table
% -------------------------------
meta = table( ...
    caseIDList, ...
    sliceIdxList, ...
    hasTumorList, ...
    brainPixelCountList, ...
    tumorPixelCountList, ...
    'VariableNames', { ...
        'CaseID', ...
        'SliceIndex', ...
        'HasTumor', ...
        'BrainPixelCount', ...
        'TumorPixelCount' ...
    });

%% -------------------------------
% 7. Summary
% -------------------------------
fprintf('\n===== Dataset Summary =====\n');
fprintf('Total samples: %d\n', numSamples);
fprintf('Image tensor size: %s\n', mat2str(size(images)));
fprintf('Mask tensor size:  %s\n', mat2str(size(masks)));
fprintf('Tumor slices: %d\n', sum(meta.HasTumor));
fprintf('Non-tumor slices: %d\n', sum(~meta.HasTumor));

%% -------------------------------
% 8. Save dataset
% -------------------------------
save(outputFile, 'images', 'masks', 'meta', ...
    'modality', 'resizeTo', 'minBrainPixels', ...
    'minTumorPixels', 'keepMode', 'maxNonTumorPerCase', '-v7.3');

fprintf('Saved dataset to: %s\n', outputFile);

%% -------------------------------
% 9. Quick visualisation
% -------------------------------
rng(1);
idx = randi(numSamples);     % Choosing one random sample 

% to check the slice loaded correctly and the mask aligns correctly
figure;
subplot(1,2,1);
imshow(images(:,:,1,idx), []);
title(sprintf('Image | %s | slice %d', meta.CaseID{idx}, meta.SliceIndex(idx)));

subplot(1,2,2);
imshow(labeloverlay(mat2gray(images(:,:,1,idx)), masks(:,:,1,idx)));
title(sprintf('Mask overlay | tumor=%d', meta.HasTumor(idx)));

%% ==========================================
% Local helper function
% ==========================================
function volNorm = normalize_mri_volume(vol)
% Normalize MRI volume using nonzero voxels only

    vol = double(vol);
    brainMask = vol > 0;

    if nnz(brainMask) == 0
        warning('Volume contains no nonzero voxels. Returning zeros.');
        volNorm = zeros(size(vol), 'single');
        return;
    end

    %  computes mean and std only on nonzero voxels
    mu = mean(vol(brainMask));
    sigma = std(vol(brainMask));

    if sigma < eps
        warning('Near-zero standard deviation in volume. Returning zeros.');
        volNorm = zeros(size(vol), 'single');
        return;
    end

    volNorm = (vol - mu) / sigma;      % Z-score normalisation

    % Set background to zero for stability
    volNorm(~brainMask) = 0;

    volNorm = single(volNorm);   % Convert back to single precision for efficient storage/training
end