function data = load_brats_case(caseFolder)
% ==========================================
% LOAD_BRATS_CASE
% Loads a single BraTS case (MRI + mask)
%
% INPUT:
%   caseFolder -> path to one patient folder
%
% OUTPUT:
%   data -> struct with fields containing MRI volumes:
%       .t1
%       .t1ce
%       .t2
%       .flair
%       .seg
%       .info
% ==========================================

fprintf('Loading BraTS case from:\n%s\n', caseFolder);

% --- Find NIfTI files ---
files = dir(fullfile(caseFolder, '*.nii*'));   % Looks for all .nii and .nii.gz files

if isempty(files)
    error('No NIfTI files found in folder.');
end

% Initialize
data = struct();

for i = 1:length(files)         % Iterates over all NIfTI files, Each file = one modality or mask
    filePath = fullfile(files(i).folder, files(i).name);
    name = lower(files(i).name);

    % Load volume
    vol = niftiread(filePath);        % loads MRI volume, Output = 3D array
    info = niftiinfo(filePath);       % Reads metadata, contains voxel size, datatype, orientation

    % Convert to double for processing(normalisation, math operations)
    vol = double(vol);           

    % Identify modality
    if contains(name, 'flair')
        data.flair = vol;            % save image
        data.info.flair = info;      % save metadata

    elseif contains(name, 't1ce')
        data.t1ce = vol;
        data.info.t1ce = info;

    elseif contains(name, 't1')
        data.t1 = vol;
        data.info.t1 = info;

    elseif contains(name, 't2')
        data.t2 = vol;
        data.info.t2 = info;

    elseif contains(name, 'seg')       % loads ground truth mask
        data.seg = vol;
        data.info.seg = info;
    end
end

% --- Check required data ---
if ~isfield(data, 'flair')
    warning('FLAIR not found.');
end

if ~isfield(data, 'seg')
    warning('Segmentation mask not found.');
end

% --- Print summary ---
fprintf('\nLoaded modalities:\n');
disp(fieldnames(data));

if isfield(data, 'flair')
    fprintf('Volume size: ');
    disp(size(data.flair));
end

end