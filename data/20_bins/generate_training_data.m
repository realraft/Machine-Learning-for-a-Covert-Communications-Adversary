%% Generate CSV features for linear vs nonlinear signals (20 FFT bins, dB scale).

%% Setup
scriptDir = fileparts(mfilename('fullpath'));
addpath(fullfile(scriptDir, '..', 'helper'));

config = get_data_params(20, scriptDir, '20_bins_training_data.csv');

featureMatrix = zeros(config.numRuns, config.numFftFeatures);
labels = zeros(config.numRuns, 1);

%% Simulations
for idx = 1:config.numRuns
    useNonlinear = idx > config.numRuns / 2;
    labels(idx) = double(useNonlinear);

    [avgPower, ~] = compute_fft_average(config.params, config.pulse, useNonlinear, config.includeNoise);
    fftPowerDb = 10 * log10(avgPower + eps);

    featureMatrix(idx, :) = fftPowerDb(config.fftSampleIdx);
end

%% Export
write_feature_table(featureMatrix, labels, config.targetFreqs, {}, config.outputFile);
