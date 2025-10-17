%% Generate CSV features for linear vs nonlinear signals (20 FFT bins + energy metrics, dB scale).

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
numRuns = 1000; % choose an even value for a balanced dataset
outputName = '20_bins_energy_training_data.csv';

scriptDir = fileparts(mfilename('fullpath'));
addpath(fullfile(scriptDir, '..', 'tools'));

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
freqAxis = (0:chunk-1) * (fs / chunk);

numFftFeatures = 20;
maxAnalysisFreq = 5;
targetFreqs = linspace(0, maxAnalysisFreq, numFftFeatures);

extraFeatureNames = { ...
    'total_power', ...
    'max_power', ...
    'crest_factor', ...
    'regrowth_power', ...
    'outer_power_ratio', ...
    'shoulder_power_ratio', ...
    'adjacent_channel_power_ratio'};
numExtraFeatures = numel(extraFeatureNames);

featureMatrix = zeros(numRuns, numFftFeatures + numExtraFeatures);
labels = zeros(numRuns, 1);

outputFile = fullfile(scriptDir, outputName);

analysisMask = freqAxis <= maxAnalysisFreq;
analysisFreqs = freqAxis(analysisMask);

% Simulations -------------------------------------------------------------
for idx = 1:numRuns
    useNonlinear = idx > numRuns/2;
    labels(idx) = double(useNonlinear);

    [avgPower, yTime] = compute_fft_average(params, pulse, useNonlinear, includeNoise);
    fftPowerDb = 10 * log10(avgPower + eps);

    fftFeatures = interp1(freqAxis, fftPowerDb, targetFreqs, 'linear');

    analysisPower = avgPower(analysisMask);
    powerSum = sum(analysisPower) + eps;

    mainMask = analysisFreqs <= 1;
    shoulderMask = analysisFreqs > 1 & analysisFreqs <= 2.5;
    outerMask = analysisFreqs > 2.5;
    adjacentMask = analysisFreqs > 1 & analysisFreqs <= 2;
    regrowthMask = analysisFreqs >= 3 & analysisFreqs <= maxAnalysisFreq;

    mainPower = sum(analysisPower(mainMask)) + eps;
    shoulderPower = sum(analysisPower(shoulderMask)) + eps;
    outerPower = sum(analysisPower(outerMask)) + eps;
    adjacentPower = sum(analysisPower(adjacentMask)) + eps;
    regrowthPower = sum(analysisPower(regrowthMask)) + eps;

    rmsVal = sqrt(mean(yTime.^2) + eps);
    crestFactor = max(abs(yTime)) / rmsVal;

    energyFeatures = [ ...
        powerSum, ...
        max(analysisPower), ...
        crestFactor, ...
        regrowthPower, ...
        outerPower / mainPower, ...
        shoulderPower / mainPower, ...
        adjacentPower / mainPower];

    featureMatrix(idx, :) = [fftFeatures, energyFeatures];
end

% Export ------------------------------------------------------------------
fftFeatureNames = arrayfun(@(k) sprintf('fft_%02d', k), 1:numFftFeatures, 'UniformOutput', false);
featureNames = [fftFeatureNames, extraFeatureNames];

featureTable = array2table(featureMatrix, 'VariableNames', featureNames);
featureTable.label = labels;

writetable(featureTable, outputFile);
fprintf('Saved %d rows to %s\n', numRuns, outputFile);
