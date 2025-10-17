%% Optimize amplitude and cubic nonlinearity parameters for threshold detection.
%% last run a = 3.2801, a3 = 0.2992

% Editable parameters -----------------------------------------------------
Nsym = 1000;
Nfft = 100;
beta = 0.25;
span = 10;
osr = 16;
Ts = 1;
avoid = 10;
aInitial = 1.0;
a1 = 1.0;
a3Initial = -0.05;
noiseVar = 0.1;
includeNoise = true;
targetError = 0.4;
rngSeed = 2025;
numTrials = 3;
detectionOffset = 0.02;
detectionWidth = 0.1;
enableCoarseSearch = true;
aGrid = linspace(0.5, 3.0, 8);
a3Grid = linspace(-0.3, 0.3, 13);

% Derived setup -----------------------------------------------------------
T = Ts / osr;
fs = 1 / T;
chunk = Nsym * osr;

pulse = rcosdesign(beta, span, osr, 'sqrt');

config = struct('Nsym', Nsym, ...
                'Nfft', Nfft, ...
                'osr', osr, ...
                'avoid', avoid, ...
                'a', aInitial, ...
                'a1', a1, ...
                'a3', a3Initial, ...
                'noiseVar', noiseVar, ...
                'includeNoise', includeNoise, ...
                'numTrials', numTrials, ...
                'rngSeed', rngSeed, ...
                'detectionOffset', detectionOffset, ...
                'detectionWidth', detectionWidth);

objective = @(vec) thresholdObjective(vec, config, pulse, targetError);

seedVec = [aInitial; a3Initial];
seedLoss = objective(seedVec);

if enableCoarseSearch
    bestVec = seedVec;
    bestLoss = seedLoss;

    for aCandidate = aGrid
        for a3Candidate = a3Grid
            candidateVec = [aCandidate; a3Candidate];
            candidateLoss = objective(candidateVec);

            if candidateLoss < bestLoss
                bestLoss = candidateLoss;
                bestVec = candidateVec;
            end
        end
    end

    fprintf('Coarse grid seed: a = %.4f, a3 = %.4f (loss %.4g)\n', bestVec(1), bestVec(2), bestLoss);
else
    bestVec = seedVec;
end

options = optimset('Display', 'iter', 'TolX', 1e-3, 'TolFun', 1e-3, ...
                   'MaxIter', 150, 'MaxFunEvals', 300);

[optVec, fval, exitflag, output] = fminsearch(objective, bestVec, options);

config.a = optVec(1);
config.a3 = optVec(2);

[errorRate, detectionRate, nonlinearPSD, linearPSD] = evaluateSpectra(config, pulse);
binaryDetectionRate = mean(nonlinearPSD > linearPSD);

fprintf('Optimization finished with exit flag %d after %d iterations.\n', exitflag, output.iterations);
fprintf('Final objective value %.4g\n', fval);
fprintf('Optimal a = %.4f, a3 = %.4f\n', config.a, config.a3);
fprintf('Detection rate (smoothed) = %.4f, error rate = %.4f (target %.4f)\n', detectionRate, errorRate, targetError);
fprintf('Detection rate (binary mask) = %.4f\n', binaryDetectionRate);

%% Local helper functions -------------------------------------------------
function loss = thresholdObjective(candidate, baseConfig, pulse, targetError)
    config = baseConfig;
    config.a = candidate(1);
    config.a3 = candidate(2);

    if config.a <= 0 || ~isfinite(config.a) || ~isfinite(config.a3)
        loss = inf;
        return;
    end

    errorRate = evaluateError(config, pulse);
    loss = (errorRate - targetError) ^ 2;
end

function errorRate = evaluateError(config, pulse)
    errorRate = evaluateSpectra(config, pulse);
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
