%% Generate CSV features for linear vs nonlinear signals (20 FFT bins + distortion metrics, dB scale).

%% Setup
scriptDir = fileparts(mfilename('fullpath'));
addpath(fullfile(scriptDir, '..', 'helper'));

config = get_data_params(20, scriptDir, '20_bins_distortion_training_data.csv');

extraFeatureNames = {
    'harmonic_distortion_ratio', ...
    'harmonic_distortion_db', ...
    'intermod_power_ratio', ...
    'intermod_power_db', ...
    'acpr_db', ...
    'regrowth_power_db', ...
    'nonlinear_residual_ratio'};

numExtraFeatures = numel(extraFeatureNames);

featureMatrix = zeros(config.numRuns, config.numFftFeatures + numExtraFeatures);
labels = zeros(config.numRuns, 1);

masks = get_frequency_masks(config.analysisFreqs);

%% Simulations
for idx = 1:config.numRuns
    useNonlinear = idx > config.numRuns / 2;
    labels(idx) = double(useNonlinear);

    [avgPower, ~] = compute_fft_average(config.params, config.pulse, useNonlinear, config.includeNoise);
    fftPowerDb = 10 * log10(avgPower + eps);

    fftFeatures = fftPowerDb(config.fftSampleIdx);

    analysisPower = avgPower(config.analysisMask);

    mainPower = sum(analysisPower(masks.main)) + eps;
    adjacentPower = sum(analysisPower(masks.adjacent)) + eps;
    intermodPower = sum(analysisPower(masks.intermod)) + eps;
    harmonicPower = sum(analysisPower(masks.harmonic)) + eps;
    regrowthPower = sum(analysisPower(masks.regrowth)) + eps;
    residualPower = sum(analysisPower(~masks.main)) + eps;

    acprDb = 10 * log10(adjacentPower / mainPower);
    harmonicRatio = harmonicPower / mainPower;
    intermodRatio = intermodPower / mainPower;
    residualRatio = residualPower / mainPower;

    distortionFeatures = [...
        harmonicRatio,...
        10 * log10(harmonicRatio + eps),...
        intermodRatio,...
        10 * log10(intermodRatio + eps),...
        acprDb,...
        10 * log10(regrowthPower + eps),...
        residualRatio];

    featureMatrix(idx, :) = [fftFeatures, distortionFeatures];
end

%% Export
write_feature_table(featureMatrix, labels, config.targetFreqs, extraFeatureNames, config.outputFile);
