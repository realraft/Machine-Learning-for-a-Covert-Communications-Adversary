%% Generate CSV features for linear vs nonlinear signals (60 FFT bins + additional metrics, dB scale).

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
numRuns = 5000; % choose an even value for a balanced dataset
outputName = 'test_data.csv';

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

numFftFeatures = 60;
maxAnalysisFreq = 5;
targetFreqs = linspace(0, maxAnalysisFreq, numFftFeatures);

extraFeatureNames = { ...
    'total_power', ...
    'max_power', ...
    'crest_factor', ...
    'spectral_entropy', ...
    'spectral_centroid_norm', ...
    'spectral_skewness', ...
    'spectral_kurtosis', ...
    'shoulder_power_ratio', ...
    'outer_power_ratio', ...
    'spectral_flatness', ...
    'acpr_db', ...
    'harmonic_distortion_ratio', ...
    'regrowth_power_db'};
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
    psd = analysisPower / powerSum;

    totalPower = powerSum;
    maxPower = max(analysisPower);

    rmsVal = sqrt(mean(yTime.^2) + eps);
    crestFactor = max(abs(yTime)) / rmsVal;

    spectralEntropy = -sum(psd .* log2(psd + eps));

    centroid = sum(analysisFreqs .* analysisPower) / powerSum;
    centroidNorm = centroid / (max(analysisFreqs) + eps);

    freqDiff = analysisFreqs - centroid;
    spectralStd = sqrt(sum((freqDiff .^ 2) .* psd) + eps);
    spectralSkewness = sum((freqDiff .^ 3) .* psd) / (spectralStd ^ 3 + eps);
    spectralKurtosis = sum((freqDiff .^ 4) .* psd) / (spectralStd ^ 4 + eps);

    mainMask = analysisFreqs <= 1;
    shoulderMask = analysisFreqs > 1 & analysisFreqs <= 2.5;
    outerMask = analysisFreqs > 2.5;
    adjacentMask = analysisFreqs > 1 & analysisFreqs <= 2;
    harmonicMask = analysisFreqs >= 2 & analysisFreqs <= maxAnalysisFreq;
    regrowthMask = analysisFreqs >= 3 & analysisFreqs <= maxAnalysisFreq;

    mainPower = sum(analysisPower(mainMask)) + eps;
    shoulderPower = sum(analysisPower(shoulderMask)) + eps;
    outerPower = sum(analysisPower(outerMask)) + eps;
    adjacentPower = sum(analysisPower(adjacentMask)) + eps;
    harmonicPower = sum(analysisPower(harmonicMask)) + eps;
    regrowthPower = sum(analysisPower(regrowthMask)) + eps;

    shoulderRatio = shoulderPower / mainPower;
    outerRatio = outerPower / mainPower;

    spectralFlatness = exp(mean(log(analysisPower + eps))) / (mean(analysisPower) + eps);

    acprDb = 10 * log10(adjacentPower / mainPower);
    hdr = harmonicPower / mainPower;
    regrowthPowerDb = 10 * log10(regrowthPower);

    extraFeatures = [ ...
        totalPower, ...
        maxPower, ...
        crestFactor, ...
        spectralEntropy, ...
        centroidNorm, ...
        spectralSkewness, ...
        spectralKurtosis, ...
        shoulderRatio, ...
        outerRatio, ...
        spectralFlatness, ...
        acprDb, ...
        hdr, ...
        regrowthPowerDb];

    featureMatrix(idx, :) = [fftFeatures, extraFeatures];
end

% Export ------------------------------------------------------------------
fftFeatureNames = arrayfun(@(k) sprintf('fft_%02d', k), 1:numFftFeatures, 'UniformOutput', false);
featureNames = [fftFeatureNames, extraFeatureNames];

featureTable = array2table(featureMatrix, 'VariableNames', featureNames);
featureTable.label = labels;

writetable(featureTable, outputFile);
fprintf('Saved %d rows to %s\n', numRuns, outputFile);
