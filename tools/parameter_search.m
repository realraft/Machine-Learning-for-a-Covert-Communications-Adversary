%% Gradient-based search for linear versus nonlinear discrimination.

%% Parameter Initialization
scriptDir = fileparts(mfilename('fullpath'));
if isempty(scriptDir)
    scriptDir = pwd;
end

addpath(fullfile(scriptDir, '..', 'data', 'helper'));

config = get_data_params(60, scriptDir, 'parameter_search_placeholder.csv');
analysisMask = (config.freqAxis >= config.minAnalysisFreq) & (config.freqAxis <= config.maxAnalysisFreq);
numTrials = 1000;

targetPercent = 60;
tolerance = 2;
maxIterations = 20;
baseSeed = 104729;

searchParams = config.params;
bestParams = searchParams;
bestPercent = NaN;
bestIteration = 0;

stepSizes = struct('a', 0.05, 'a3', 0.1, 'noiseVar', 2);
learningRates = struct('a', 0.25, 'a3', 0.1, 'noiseVar', 0.02);
percentHistory = zeros(maxIterations, 1);

%% Gradient Search Loop
for iter = 1:maxIterations
    iterSeed = baseSeed + (iter - 1) * 1000;
    currentPercent = compute_threshold_percent(searchParams, config.pulse, analysisMask, numTrials, iterSeed);
    percentHistory(iter) = currentPercent;

    fprintf('Iteration %d: %.2f%%%% linear > nonlinear (target %.2f%%%%)\n', iter, currentPercent, targetPercent);

    bestParams = searchParams;
    bestPercent = currentPercent;
    bestIteration = iter;

    if abs(currentPercent - targetPercent) <= tolerance
        break;
    end

    gradientValues = struct('a', 0, 'a3', 0, 'noiseVar', 0);
    gradientFields = fieldnames(gradientValues);

    for idx = 1:numel(gradientFields)
        fieldName = gradientFields{idx};
        stepAmount = stepSizes.(fieldName);
        perturbedParams = searchParams;
        perturbedParams.(fieldName) = perturbedParams.(fieldName) + stepAmount;

        if strcmp(fieldName, 'noiseVar')
            perturbedParams.(fieldName) = max(1e-3, perturbedParams.(fieldName));
        end

        testPercent = compute_threshold_percent(perturbedParams, config.pulse, analysisMask, numTrials, iterSeed);
        gradientValues.(fieldName) = (testPercent - currentPercent) / stepAmount;
    end

    errorValue = currentPercent - targetPercent;

    deltaA = learningRates.a * errorValue * gradientValues.a;
    deltaA = max(-0.5, min(0.5, deltaA));
    searchParams.a = max(0.1, searchParams.a - deltaA);

    deltaA3 = learningRates.a3 * errorValue * gradientValues.a3;
    deltaA3 = max(-0.5, min(0.5, deltaA3));
    searchParams.a3 = min(0, max(-10, searchParams.a3 - deltaA3));

    deltaNoise = learningRates.noiseVar * errorValue * gradientValues.noiseVar;
    deltaNoise = max(-5, min(5, deltaNoise));
    searchParams.noiseVar = max(1e-3, searchParams.noiseVar - deltaNoise);
end

if bestIteration == 0
    bestIteration = maxIterations;
    lastIdx = find(percentHistory ~= 0, 1, 'last');
    if isempty(lastIdx)
        bestPercent = compute_threshold_percent(searchParams, config.pulse, analysisMask, numTrials, baseSeed);
    else
        bestPercent = percentHistory(lastIdx);
    end
    bestParams = searchParams;
end

%% Signal Generation and FFT
finalSeed = baseSeed + bestIteration * 1000 + 500;
[linearMagnitudes, nonlinearMagnitudes] = generate_fft_samples(bestParams, config.pulse, numTrials, finalSeed);
linearBand = linearMagnitudes(:, analysisMask);
nonlinearBand = nonlinearMagnitudes(:, analysisMask);
bandCount = size(linearBand, 2);
analysisCount = bandCount;

%% Threshold Test
comparisonCounts = sum((linearBand .^ 2) > (nonlinearBand .^ 2), 2);
percentages = (comparisonCounts / analysisCount) * 100;
averagedPercent = mean(percentages);
percentStd = std(percentages);

%% Final Results
fprintf('Gradient search completed after %d iterations.\n', bestIteration);
fprintf('Loop estimate: %.2f%%%% linear > nonlinear.\n', bestPercent);
fprintf('Mean threshold result: %.2f%%%% linear > nonlinear (target %.2f%%%%).\n', averagedPercent, targetPercent);
fprintf('Std of threshold result: %.2f%%%%.\n', percentStd);
fprintf('Final parameters: a=%.4f, a3=%.4f, noiseVar=%.4f.\n', bestParams.a, bestParams.a3, bestParams.noiseVar);

function percent = compute_threshold_percent(paramsStruct, pulseShape, analysisMask, numTrials, baseSeed)
%COMPUTE_THRESHOLD_PERCENT Evaluate percentage where linear power exceeds nonlinear power.
    if nargin < 5
        baseSeed = [];
    end

    if ~isempty(baseSeed)
        originalState = rng;
        cleanupObj = onCleanup(@() rng(originalState)); %#ok<NASGU>
    end

    numPoints = nnz(analysisMask);
    trialPercentages = zeros(numTrials, 1);

    for trial = 1:numTrials
        if ~isempty(baseSeed)
            rng(baseSeed + trial);
        end
        [linearSpectrum, ~] = compute_fft_average(paramsStruct, pulseShape, false, true);

        if ~isempty(baseSeed)
            rng(baseSeed + trial);
        end
        [nonlinearSpectrum, ~] = compute_fft_average(paramsStruct, pulseShape, true, true);

        linearPower = linearSpectrum(analysisMask);
        nonlinearPower = nonlinearSpectrum(analysisMask);

        trialPercentages(trial) = sum(linearPower > nonlinearPower) / numPoints * 100;
    end

    percent = mean(trialPercentages);
end

function [linearMagnitudes, nonlinearMagnitudes] = generate_fft_samples(paramsStruct, pulseShape, numTrials, baseSeed)
%GENERATE_FFT_SAMPLES Generate FFT magnitudes for linear and nonlinear cases.
    if nargin < 4
        baseSeed = [];
    end

    if ~isempty(baseSeed)
        originalState = rng;
        cleanupObj = onCleanup(@() rng(originalState)); %#ok<NASGU>
    end

    chunkLength = paramsStruct.Nsym * paramsStruct.osr;
    linearMagnitudes = zeros(numTrials, chunkLength);
    nonlinearMagnitudes = zeros(numTrials, chunkLength);

    for trial = 1:numTrials
        if ~isempty(baseSeed)
            rng(baseSeed + trial);
        end
        [linearSpectrum, ~] = compute_fft_average(paramsStruct, pulseShape, false, true);
        linearMagnitudes(trial, :) = sqrt(linearSpectrum);

        if ~isempty(baseSeed)
            rng(baseSeed + trial);
        end
        [nonlinearSpectrum, ~] = compute_fft_average(paramsStruct, pulseShape, true, true);
        nonlinearMagnitudes(trial, :) = sqrt(nonlinearSpectrum);
    end
end
