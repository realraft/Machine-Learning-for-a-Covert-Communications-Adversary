%% Generate CSV features for linear vs nonlinear signals (20 FFT bins + energy metrics, dB scale).

%% Setup
scriptDir = fileparts(mfilename('fullpath'));
addpath(fullfile(scriptDir, '..', 'helper'));

config = get_data_params(20, scriptDir, '20_bins_energy_training_data.csv');

extraFeatureNames = {
    'total_power', ...
    'max_power', ...
    'crest_factor', ...
    'regrowth_power', ...
    'outer_power_ratio', ...
    'shoulder_power_ratio', ...
    'adjacent_channel_power_ratio'};

numExtraFeatures = numel(extraFeatureNames);

featureMatrix = zeros(config.numRuns, config.numFftFeatures + numExtraFeatures);
labels = zeros(config.numRuns, 1);

masks = get_frequency_masks(config.analysisFreqs);

%% Simulations
for idx = 1:config.numRuns
    useNonlinear = idx > config.numRuns / 2;
    labels(idx) = double(useNonlinear);

    [avgPower, yTime] = compute_fft_average(config.params, config.pulse, useNonlinear, config.includeNoise);
    fftPowerDb = 10 * log10(avgPower + eps);

    fftFeatures = fftPowerDb(config.fftSampleIdx);

    analysisPower = avgPower(config.analysisMask);
    stats = compute_spectral_stats(config.analysisFreqs, analysisPower);

    mainPower = sum(analysisPower(masks.main)) + eps;
    shoulderPower = sum(analysisPower(masks.shoulder)) + eps;
    outerPower = sum(analysisPower(masks.outer)) + eps;
    adjacentPower = sum(analysisPower(masks.adjacent)) + eps;
    regrowthPower = sum(analysisPower(masks.regrowth)) + eps;

    energyFeatures = [...
        stats.powerSum,...
        stats.maxPower,...
        compute_crest_factor(yTime),...
        regrowthPower,...
        outerPower / mainPower,...
        shoulderPower / mainPower,...
        adjacentPower / mainPower];

    featureMatrix(idx, :) = [fftFeatures, energyFeatures];
end

%% Export
write_feature_table(featureMatrix, labels, config.targetFreqs, extraFeatureNames, config.outputFile);
