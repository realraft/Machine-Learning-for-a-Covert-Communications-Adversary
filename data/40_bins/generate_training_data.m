%% Generate CSV features for linear vs nonlinear signals (40 FFT bins, dB scale).

% Editable parameters -----------------------------------------------------
Nsym = 1000;
Nfft = 100;
beta = 0.25;
span = 10;
osr = 16;
Ts = 1;
avoid = 10;
a = 1.4678;
a1 = 1;
a3 = -2.5261;
noiseVariance = 0.1;
includeNoise = true;
numRuns = 10; % choose an even value for a balanced dataset
outputName = '40_bins_training_data.csv';

scriptDir = fileparts(mfilename('fullpath'));
addpath(fullfile(scriptDir, '..', 'helper'));

% Setup -------------------------------------------------------------------
params = struct('Nsym', Nsym, ...
                'Nfft', Nfft, ...
                'osr', osr, ...
                'avoid', avoid, ...
                'a', a, ...
                'a1', a1, ...
                'a3', a3, ...
                'noiseVar', noiseVariance);

pulse = rcosdesign(beta, span, osr, 'sqrt');

T = Ts / osr;
fs = 1 / T;
chunk = Nsym * osr;

numFeatures = 40;
maxAnalysisFreq = 5;
referenceMaxBins = 60;
[targetFreqs, fftSampleIdx] = get_fft_reference_selection(numFeatures, fs, chunk, maxAnalysisFreq, referenceMaxBins); % align to master grid

featureMatrix = zeros(numRuns, numFeatures);
labels = zeros(numRuns, 1);

outputFile = fullfile(scriptDir, outputName);

% Simulations -------------------------------------------------------------
for idx = 1:numRuns
    useNonlinear = idx > numRuns/2;
    labels(idx) = double(useNonlinear);

    [avgPower, ~] = compute_fft_average(params, pulse, useNonlinear, includeNoise);
    fftPowerDb = 10 * log10(avgPower + eps);

    featureMatrix(idx, :) = fftPowerDb(fftSampleIdx);
end

% Export ------------------------------------------------------------------
fftFeatureNames = format_fft_feature_names(targetFreqs);
featureTable = array2table(featureMatrix, 'VariableNames', fftFeatureNames);
featureTable.label = labels;

writetable(featureTable, outputFile);
fprintf('Saved %d rows to %s\n', numRuns, outputFile);
