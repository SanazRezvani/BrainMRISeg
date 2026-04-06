%% ==========================================
% EVALUATE_UNET_2D
% Evaluate trained 2D U-Net on BraTS test set
% Matched to the simplified no-augmentation training pipeline
%
% Requirements:
%   - brats2d_dataset.mat
%   - trained_unet_2d.mat
%
% Dataset file should contain:
%   images : H x W x 1 x N
%   masks  : H x W x 1 x N
%   meta   : table with CaseID and SliceIndex
%
% Model file should contain:
%   net
%   splitInfo
%
% Outputs:
%   - results/test_slice_metrics.csv
%   - results/test_patient_metrics.csv
%   - results/evaluation_summary.mat
%   - example figures in results/figures
% ==========================================

clear; clc; close all;

%% -------------------------------
% 1. User settings
% -------------------------------
datasetFile = 'brats2d_dataset.mat';
modelFile   = 'trained_unet_2d.mat';

resultsDir = 'results';
figDir = fullfile(resultsDir, 'figures');

numExampleFigures = 6;
saveFigures = true;
showFigures = false;

if ~exist(resultsDir, 'dir')
    mkdir(resultsDir);
end

if saveFigures && ~exist(figDir, 'dir')
    mkdir(figDir);
end

%% -------------------------------
% 2. Load dataset and model
% -------------------------------
S = load(datasetFile);
M = load(modelFile);

images = S.images;
masks  = S.masks;
meta   = S.meta;

net = M.net;
splitInfo = M.splitInfo;

fprintf('Loaded dataset: %s\n', datasetFile);
fprintf('Loaded model:   %s\n', modelFile);

if ~istable(meta)
    error('meta must be a table.');
end

requiredVars = {'CaseID','SliceIndex'};
for i = 1:numel(requiredVars)
    if ~ismember(requiredVars{i}, meta.Properties.VariableNames)
        error('meta table must contain column: %s', requiredVars{i});
    end
end

if ~isfield(splitInfo, 'testPatients')
    error('splitInfo.testPatients not found in model file.');
end

%% -------------------------------
% 3. Select test set by patient
% -------------------------------
testPatients = splitInfo.testPatients;

if isempty(testPatients)
    error('No test patients found in splitInfo.');
end

testIdx = ismember(meta.CaseID, testPatients);

XTest = images(:,:,:,testIdx);
YTest = masks(:,:,:,testIdx);
metaTest = meta(testIdx, :);

numTestSlices = size(XTest, 4);

fprintf('\n=== Test Set Summary ===\n');
fprintf('Number of test patients: %d\n', numel(unique(metaTest.CaseID)));
fprintf('Number of test slices:   %d\n', numTestSlices);

if numTestSlices == 0
    error('No test slices found. Check your splitInfo and meta table.');
end

%% -------------------------------
% 4. Run inference on test set
% -------------------------------
diceScores = zeros(numTestSlices, 1);
iouScores  = zeros(numTestSlices, 1);
trueTumorPixels = zeros(numTestSlices, 1);
predTumorPixels = zeros(numTestSlices, 1);

fprintf('\nRunning predictions on test slices...\n');

for i = 1:numTestSlices
    img = XTest(:,:,:,i);
    trueMask = logical(YTest(:,:,:,i));

    predCat = semanticseg(img, net);
    predMask = predCat == 'tumor';

    diceScores(i) = compute_dice_binary(predMask, trueMask);
    iouScores(i)  = compute_iou_binary(predMask, trueMask);

    trueTumorPixels(i) = nnz(trueMask);
    predTumorPixels(i) = nnz(predMask);
end

%% -------------------------------
% 5. Build slice-level metrics table
% -------------------------------
sliceMetrics = metaTest;
sliceMetrics.Dice = diceScores;
sliceMetrics.IoU = iouScores;
sliceMetrics.TrueTumorPixels = trueTumorPixels;
sliceMetrics.PredTumorPixels = predTumorPixels;

sliceMetricsFile = fullfile(resultsDir, 'test_slice_metrics.csv');
writetable(sliceMetrics, sliceMetricsFile);

fprintf('Saved slice-level metrics to: %s\n', sliceMetricsFile);

%% -------------------------------
% 6. Build patient-level metrics table
% -------------------------------
patientIDs = unique(metaTest.CaseID);
numPatients = numel(patientIDs);

patientCaseID = strings(numPatients,1);
patientNumSlices = zeros(numPatients,1);
patientNumTumorSlices = zeros(numPatients,1);
patientMeanDice = zeros(numPatients,1);
patientStdDice = zeros(numPatients,1);
patientMedianDice = zeros(numPatients,1);
patientMeanIoU = zeros(numPatients,1);

for p = 1:numPatients
    thisID = patientIDs{p};
    idx = strcmp(metaTest.CaseID, thisID);

    patientCaseID(p) = string(thisID);
    patientNumSlices(p) = sum(idx);

    if ismember('HasTumor', metaTest.Properties.VariableNames)
        patientNumTumorSlices(p) = sum(metaTest.HasTumor(idx));
    else
        patientNumTumorSlices(p) = sum(trueTumorPixels(idx) > 0);
    end

    patientMeanDice(p) = mean(diceScores(idx));
    patientStdDice(p) = std(diceScores(idx));
    patientMedianDice(p) = median(diceScores(idx));
    patientMeanIoU(p) = mean(iouScores(idx));
end

patientMetrics = table( ...
    patientCaseID, ...
    patientNumSlices, ...
    patientNumTumorSlices, ...
    patientMeanDice, ...
    patientStdDice, ...
    patientMedianDice, ...
    patientMeanIoU, ...
    'VariableNames', { ...
        'CaseID', ...
        'NumSlices', ...
        'NumTumorSlices', ...
        'MeanDice', ...
        'StdDice', ...
        'MedianDice', ...
        'MeanIoU' ...
    });

patientMetricsFile = fullfile(resultsDir, 'test_patient_metrics.csv');
writetable(patientMetrics, patientMetricsFile);

fprintf('Saved patient-level metrics to: %s\n', patientMetricsFile);

%% -------------------------------
% 7. Print overall summary
% -------------------------------
overallMeanDice = mean(diceScores);
overallStdDice = std(diceScores);
overallMedianDice = median(diceScores);
overallMeanIoU = mean(iouScores);

fprintf('\n===== Overall Test Performance =====\n');
fprintf('Mean Dice:   %.4f\n', overallMeanDice);
fprintf('Std Dice:    %.4f\n', overallStdDice);
fprintf('Median Dice: %.4f\n', overallMedianDice);
fprintf('Mean IoU:    %.4f\n', overallMeanIoU);

%% -------------------------------
% 8. Save summary plots
% -------------------------------
fig1 = figure('Color','w');
histogram(diceScores, 20);
xlabel('Dice Score');
ylabel('Number of Slices');
title('Dice Score Distribution on Test Slices');
grid on;

if saveFigures
    saveas(fig1, fullfile(figDir, 'dice_histogram.png'));
end
if ~showFigures
    close(fig1);
end

fig2 = figure('Color','w');
bar(patientMeanDice);
xticks(1:numPatients);
xticklabels(patientMetrics.CaseID);
xtickangle(45);
ylabel('Mean Dice');
title('Mean Dice per Test Patient');
grid on;

if saveFigures
    saveas(fig2, fullfile(figDir, 'patient_mean_dice.png'));
end
if ~showFigures
    close(fig2);
end

%% -------------------------------
% 9. Save example prediction figures
% -------------------------------
exampleIndices = find(trueTumorPixels > 0);

if isempty(exampleIndices)
    exampleIndices = 1:numTestSlices;
end

numExampleFigures = min(numExampleFigures, numel(exampleIndices));
exampleIndices = exampleIndices(1:numExampleFigures);

fprintf('Saving %d example prediction figures...\n', numExampleFigures);

for k = 1:numExampleFigures
    idx = exampleIndices(k);

    img = XTest(:,:,:,idx);
    trueMask = logical(YTest(:,:,:,idx));

    predCat = semanticseg(img, net);
    predMask = predCat == 'tumor';

    thisDice = compute_dice_binary(predMask, trueMask);
    thisCase = metaTest.CaseID{idx};
    thisSlice = metaTest.SliceIndex(idx);

    fig = figure('Visible','off', 'Color','w', 'Position',[100 100 1200 350]);

    subplot(1,4,1);
    imshow(img, []);
    title(sprintf('Input\n%s | Slice %d', thisCase, thisSlice), 'Interpreter','none');

    subplot(1,4,2);
    imshow(labeloverlay(mat2gray(img), trueMask));
    title('Ground Truth');

    subplot(1,4,3);
    imshow(labeloverlay(mat2gray(img), predMask));
    title(sprintf('Prediction\nDice = %.3f', thisDice));

    subplot(1,4,4);
    errorOverlay = make_error_overlay(img, trueMask, predMask);
    imshow(errorOverlay);
    title('Error Overlay');

    if saveFigures
        outName = sprintf('example_%02d_%s_slice_%03d.png', ...
            k, thisCase, thisSlice);
        saveas(fig, fullfile(figDir, outName));
    end

    if showFigures
        set(fig, 'Visible', 'on');
    else
        close(fig);
    end
end

%% -------------------------------
% 10. Save summary MAT file
% -------------------------------
summaryFile = fullfile(resultsDir, 'evaluation_summary.mat');
save(summaryFile, ...
    'sliceMetrics', 'patientMetrics', ...
    'diceScores', 'iouScores', ...
    'overallMeanDice', 'overallStdDice', ...
    'overallMedianDice', 'overallMeanIoU', ...
    '-v7.3');

fprintf('Saved evaluation summary to: %s\n', summaryFile);
fprintf('\nEvaluation complete.\n');

%% ==========================================
% Local helper functions
% ==========================================

function d = compute_dice_binary(predMask, trueMask)
% Compute Dice score for binary masks
    predMask = logical(predMask);
    trueMask = logical(trueMask);

    intersection = nnz(predMask & trueMask);
    totalPixels = nnz(predMask) + nnz(trueMask);

    if totalPixels == 0
        d = 1.0;
    else
        d = 2 * intersection / totalPixels;
    end
end

function iou = compute_iou_binary(predMask, trueMask)
% Compute Intersection-over-Union for binary masks
    predMask = logical(predMask);
    trueMask = logical(trueMask);

    intersection = nnz(predMask & trueMask);
    unionSet = nnz(predMask | trueMask);

    if unionSet == 0
        iou = 1.0;
    else
        iou = intersection / unionSet;
    end
end

function rgb = make_error_overlay(img, trueMask, predMask)
% Create RGB error overlay:
%   TP = green
%   FN = red
%   FP = yellow

    img = mat2gray(img);
    rgb = repmat(img, [1 1 3]);

    tp = predMask & trueMask;
    fn = ~predMask & trueMask;
    fp = predMask & ~trueMask;

    red = rgb(:,:,1);
    green = rgb(:,:,2);
    blue = rgb(:,:,3);

    red(fn) = 1.0;
    red(fp) = 1.0;

    green(tp) = 1.0;
    green(fp) = 1.0;

    blue(tp) = 0.0;
    blue(fn) = 0.0;
    blue(fp) = 0.0;

    rgb(:,:,1) = red;
    rgb(:,:,2) = green;
    rgb(:,:,3) = blue;
end