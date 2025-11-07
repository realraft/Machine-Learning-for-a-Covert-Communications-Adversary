%% Optimize cubic nonlinearity and noise variance parameters for threshold detection.

%% Setup helper path
scriptDir = fileparts(mfilename('fullpath'));
helperDir = fullfile(scriptDir, '..', 'data', 'helper');
if exist(helperDir, 'dir')
    addpath(helperDir);
end

%% Parameter configuration
Nsym = 1000;
Nfft = 100;
beta = 0.25;
span = 10;
osr = 16;
avoid = 10;
aFixed = 1.5;
a1 = 1;
a3Initial = -2.5;
noiseInitial = 1;
includeNoise = true;
targetError = 0.4;
rngSeed = 2025;
numTrials = 3;
detectionOffset = 0.02;
detectionWidth = 0.1;
enableCoarseSearch = true;
a3Window = 1;
a3LowerBound = a3Initial - a3Window;
a3UpperBound = a3Initial + a3Window;
a3Grid = linspace(a3LowerBound, a3UpperBound, 13);
noiseLowerBound = noiseInitial;
noiseGrid = linspace(noiseLowerBound + 0.05, noiseLowerBound + 0.8, 13);

if noiseInitial <= noiseLowerBound
    noiseInitial = noiseLowerBound + 0.05;
end

if a3Initial <= a3LowerBound
    a3Initial = a3LowerBound + 0.01;
elseif a3Initial >= a3UpperBound
    a3Initial = a3UpperBound - 0.01;
end
logNoiseInitial = log(max(noiseInitial - noiseLowerBound, 1e-6));

pulse = rcosdesign(beta, span, osr, 'sqrt');

config = struct('Nsym', Nsym, ...
                'Nfft', Nfft, ...
                'osr', osr, ...
                'avoid', avoid, ...
                'a', aFixed, ...
                'a1', a1, ...
                'a3', a3Initial, ...
                'noiseVar', noiseInitial, ...
                'includeNoise', includeNoise, ...
                'numTrials', numTrials, ...
                'rngSeed', rngSeed, ...
                'detectionOffset', detectionOffset, ...
                'detectionWidth', detectionWidth);

objective = @(vec) thresholdObjective(vec, config, pulse, targetError, noiseLowerBound);

seedVec = [a3Initial; logNoiseInitial];
bestVec = seedVec;
bestLoss = objective(seedVec);

if enableCoarseSearch
    for a3Candidate = a3Grid
        for noiseCandidate = noiseGrid
            if noiseCandidate <= noiseLowerBound
                continue;
            end

            candidateVec = [a3Candidate; log(max(noiseCandidate - noiseLowerBound, 1e-6))];
            candidateLoss = objective(candidateVec);

            if candidateLoss < bestLoss
                bestLoss = candidateLoss;
                bestVec = candidateVec;
            end
        end
    end

    fprintf('Coarse grid seed: a3 = %.4f, noise variance = %.4f (loss %.4g)\n', bestVec(1), noiseLowerBound + exp(bestVec(2)), bestLoss);
end

options = optimset('Display', 'iter', 'TolX', 1e-3, 'TolFun', 1e-3, ...
                   'MaxIter', 150, 'MaxFunEvals', 300);

[optVec, fval, exitflag, output] = fminsearch(objective, bestVec, options);

config.a3 = optVec(1);
config.noiseVar = noiseLowerBound + exp(optVec(2));

[errorRate, detectionRate, nonlinearPSD, linearPSD] = evaluateSpectra(config, pulse);
binaryDetectionRate = mean(nonlinearPSD > linearPSD);

fprintf('Optimization finished with exit flag %d after %d iterations.\n', exitflag, output.iterations);
fprintf('Final objective value %.4g\n', fval);
fprintf('Optimal a3 = %.4f, noise variance = %.4f\n', config.a3, config.noiseVar);
fprintf('Detection rate (smoothed) = %.4f, error rate = %.4f (target %.4f)\n', detectionRate, errorRate, targetError);
fprintf('Detection rate (binary mask) = %.4f\n', binaryDetectionRate);

%% Local helper functions -------------------------------------------------
function loss = thresholdObjective(candidate, baseConfig, pulse, targetError, noiseLowerBound)
    config = baseConfig;
    config.a3 = candidate(1);
    config.noiseVar = noiseLowerBound + exp(candidate(2));

    if ~isfinite(config.a3) || ~isfinite(candidate(2))
        loss = inf;
        return;
    end

    [errorRate, ~] = evaluateSpectra(config, pulse);
    loss = (errorRate - targetError) ^ 2;
end

function [errorRate, detectionRate, nonlinearPSD, linearPSD] = evaluateSpectra(config, pulse)
    chunkSize = config.Nsym * config.osr;
    linearPSD = zeros(1, chunkSize);
    nonlinearPSD = zeros(1, chunkSize);

    for trial = 1:config.numTrials
        seed = config.rngSeed + trial - 1;

        rng(seed);
        linearPSD = linearPSD + compute_fft_average(config, pulse, false, config.includeNoise);

        rng(seed);
        nonlinearPSD = nonlinearPSD + compute_fft_average(config, pulse, true, config.includeNoise);
    end

    linearPSD = linearPSD / config.numTrials;
    nonlinearPSD = nonlinearPSD / config.numTrials;

    ratioDiff = (nonlinearPSD - linearPSD) ./ max(linearPSD, eps);
    excess = ratioDiff - config.detectionOffset;
    scaled = excess / config.detectionWidth;
    clamped = min(max(scaled, 0), 1);

    detectionRate = mean(clamped);
    detectionRate = min(max(detectionRate, 0), 1);
    errorRate = 1 - detectionRate;
end
