%% Generate CSV features for linear vs nonlinear signals (20 FFT bins + shape metrics, dB scale).

%% Setup
scriptDir = fileparts(mfilename('fullpath'));
addpath(fullfile(scriptDir, '..', 'helper'));

config = get_data_params(20, scriptDir, '20_bins_shape_training_data.csv');

extraFeatureNames = {
    'spectral_centroid_norm', ...
    'spectral_entropy', ...
    'spectral_skewness', ...
    'spectral_kurtosis', ...
    'spectral_flatness'};

numExtraFeatures = numel(extraFeatureNames);

featureMatrix = zeros(config.numRuns, config.numFftFeatures + numExtraFeatures);
labels = zeros(config.numRuns, 1);

%% Simulations
for idx = 1:config.numRuns
    useNonlinear = idx > config.numRuns / 2;
    labels(idx) = double(useNonlinear);

    [avgPower, ~] = compute_fft_average(config.params, config.pulse, useNonlinear, config.includeNoise);
    fftPowerDb = 10 * log10(avgPower + eps);

    fftFeatures = fftPowerDb(config.fftSampleIdx);

    analysisPower = avgPower(config.analysisMask);
    stats = compute_spectral_stats(config.analysisFreqs, analysisPower);

    shapeFeatures = [...
        stats.centroidNorm,...
        stats.entropy,...
        stats.skewness,...
        stats.kurtosis,...
        stats.flatness];

    featureMatrix(idx, :) = [fftFeatures, shapeFeatures];
end

%% Export
write_feature_table(featureMatrix, labels, config.targetFreqs, extraFeatureNames, config.outputFile);
