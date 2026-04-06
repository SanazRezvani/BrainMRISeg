%% ==========================================
% TRAIN_UNET_2D
% Train a small 2D U-Net on BraTS slice dataset
%
% Requirements:
%   - brats2d_dataset.mat created by build_slice_dataset.m
%   - variables in file:
%         images : H x W x 1 x N  (single)
%         masks  : H x W x 1 x N  (logical)
%         meta   : table with CaseID column
%
% Output:
%   - trained network
%   - training info
%   - patient split info
% ==========================================

clear; clc; close all;

%% -------------------------------
% 1. User settings
% -------------------------------
datasetFile = 'brats2d_dataset.mat';
outputModelFile = 'trained_unet_2d.mat';

rng(1);                     % reproducibility
trainRatio = 0.70;
valRatio   = 0.15;
testRatio  = 0.15;

initialLearnRate = 1e-3;
maxEpochs = 5;
miniBatchSize = 16;

%% -------------------------------
% 2. Load dataset
% -------------------------------
S = load(datasetFile);

images = S.images;
masks  = S.masks;
meta   = S.meta;

fprintf('Loaded dataset from %s\n', datasetFile);
fprintf('Images size: %s\n', mat2str(size(images)));
fprintf('Masks size:  %s\n', mat2str(size(masks)));
fprintf('Number of samples: %d\n', size(images,4));

if size(images,4) ~= size(masks,4)
    error('Number of image slices and mask slices does not match.');
end

if ~istable(meta)
    error('meta must be a table.');
end

if ~ismember('CaseID', meta.Properties.VariableNames)
    error('meta table must contain a CaseID column.');
end

%% -------------------------------
% 3. Split by patient
% -------------------------------
allPatients = unique(meta.CaseID);
numPatients = numel(allPatients);

fprintf('Number of unique patients: %d\n', numPatients);

allPatients = allPatients(randperm(numPatients));       % Shuffling patients to avoid bias

nTrain = max(1, floor(trainRatio * numPatients));
nVal   = max(1, floor(valRatio   * numPatients));
nTest  = numPatients - nTrain - nVal;

% Ensure at least 1 patient in test when possible
if nTest < 1 && numPatients >= 3
    nTest = 1;
    if nTrain > nVal
        nTrain = nTrain - 1;
    else
        nVal = nVal - 1;
    end
end

trainPatients = allPatients(1:nTrain);
valPatients   = allPatients(nTrain+1 : nTrain+nVal);
testPatients  = allPatients(nTrain+nVal+1 : end);

fprintf('\n=== Patient Split ===\n');
fprintf('Train patients: %d\n', numel(trainPatients));
fprintf('Val patients:   %d\n', numel(valPatients));
fprintf('Test patients:  %d\n', numel(testPatients));

disp('Train patient IDs:'); disp(trainPatients');
disp('Val patient IDs:');   disp(valPatients');
disp('Test patient IDs:');  disp(testPatients');

% Slice indices for each split
trainIdx = ismember(meta.CaseID, trainPatients);
valIdx   = ismember(meta.CaseID, valPatients);
testIdx  = ismember(meta.CaseID, testPatients);

XTrain = images(:,:,:,trainIdx);
YTrain = masks(:,:,:,trainIdx);

XVal = images(:,:,:,valIdx);
YVal = masks(:,:,:,valIdx);

XTest = images(:,:,:,testIdx);
YTest = masks(:,:,:,testIdx);

fprintf('\n=== Slice Split ===\n');
fprintf('Train slices: %d\n', size(XTrain,4));
fprintf('Val slices:   %d\n', size(XVal,4));
fprintf('Test slices:  %d\n', size(XTest,4));

if isempty(XTrain) || isempty(XVal)
    error('Training or validation set is empty. Add more patients or adjust split ratios.');
end

%% -------------------------------
% 4. Convert masks to categorical
% -------------------------------
classNames = ["background", "tumor"];
labelIDs = [0 1];

YTrainCat = masks_to_categorical(YTrain, classNames);
YValCat   = masks_to_categorical(YVal, classNames);
YTestCat  = masks_to_categorical(YTest, classNames); %#ok<NASGU>

%% -------------------------------
% 5. Create datastores (no augmentation)
% -------------------------------
dsXTrain = arrayDatastore(XTrain, 'IterationDimension', 4);
dsYTrain = arrayDatastore(YTrainCat, 'IterationDimension', 4);

dsXVal = arrayDatastore(XVal, 'IterationDimension', 4);
dsYVal = arrayDatastore(YValCat, 'IterationDimension', 4);

trainDS = combine(dsXTrain, dsYTrain);
valDS   = combine(dsXVal, dsYVal);

%% -------------------------------
% 6. Define a small 2D U-Net
% -------------------------------
imageSize = [size(XTrain,1), size(XTrain,2), size(XTrain,3)];     % imageSize = [H W C]
numClasses = numel(classNames);                                   % numClasses = 2
encoderDepth = 3;

lgraph = unetLayers(imageSize, numClasses, 'EncoderDepth', encoderDepth);

pxLayer = pixelClassificationLayer( ...
    'Name', 'labels', ...
    'Classes', classNames);                % Defining output layer

lgraph = replaceLayer(lgraph, 'Segmentation-Layer', pxLayer);         % Customising final classification layer

%% -------------------------------
% 7. Training options
% -------------------------------
validationFrequency = max(1, floor(size(XTrain,4) / miniBatchSize));

options = trainingOptions('adam', ...
    'InitialLearnRate', initialLearnRate, ...
    'MaxEpochs', maxEpochs, ...
    'MiniBatchSize', miniBatchSize, ...
    'Shuffle', 'every-epoch', ...
    'ValidationData', valDS, ...
    'ValidationFrequency', validationFrequency, ...
    'Verbose', true, ...
    'Plots', 'training-progress');

%% -------------------------------
% 8. Train network
% -------------------------------
fprintf('\nStarting training...\n');

net = trainNetwork(trainDS, lgraph, options);

fprintf('Training complete.\n');

%% -------------------------------
% 9. Quick validation preview
% -------------------------------
numValToShow = min(3, size(XVal,4));

for i = 1:numValToShow
    img = XVal(:,:,:,i);
    pred = semanticseg(img, net);

    predMask = pred == 'tumor';
    trueMask = YVal(:,:,:,i);

    figure;
    subplot(1,3,1);
    imshow(img, []);
    title('Input');

    subplot(1,3,2);
    imshow(labeloverlay(mat2gray(img), trueMask));
    title('Ground Truth');

    subplot(1,3,3);
    imshow(labeloverlay(mat2gray(img), predMask));
    title('Prediction');
end

%% -------------------------------
% 10. Save model and split info
% -------------------------------
splitInfo = struct();
splitInfo.trainPatients = trainPatients;
splitInfo.valPatients   = valPatients;
splitInfo.testPatients  = testPatients;

trainingConfig = struct();
trainingConfig.initialLearnRate = initialLearnRate;
trainingConfig.maxEpochs = maxEpochs;
trainingConfig.miniBatchSize = miniBatchSize;
trainingConfig.trainRatio = trainRatio;
trainingConfig.valRatio = valRatio;
trainingConfig.testRatio = testRatio;
trainingConfig.augmentationUsed = false;

save(outputModelFile, ...
    'net', 'options', 'splitInfo', 'trainingConfig', ...
    'classNames', 'labelIDs', '-v7.3');

fprintf('\nSaved trained model to: %s\n', outputModelFile);

%% ==========================================
% Local helper function
% ==========================================
function YCat = masks_to_categorical(YLogical, classNames)
% Convert logical masks HxWx1xN into categorical labels
% background = false, tumor = true

    [H,W,~,N] = size(YLogical);
    YCat = categorical(zeros(H,W,1,N), [0 1], classNames);

    for k = 1:N
        thisMask = YLogical(:,:,:,k);
        thisMask = uint8(thisMask);  % 0 / 1
        YCat(:,:,:,k) = categorical(thisMask, [0 1], classNames);
    end
end