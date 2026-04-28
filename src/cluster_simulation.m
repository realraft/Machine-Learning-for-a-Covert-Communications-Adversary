% Parameters
nSym = 100; % Number of symbols
nFFT = 100; % Number of fft to average over
beta = 0.25; % Rolloff factor
span = 5; % Number of symbols for srrc
osr = 16; % Oversampling rate
a1 = 1; % Linear coefficient
a3 = linspace(-0.05, -0.2, 100); % Nonlinear cubic coefficient sweep
noiseRatio = 0.75; % What ratio of total linear power is used to determine noise variance
runs = 150; % Number of independent runs for dataset generation
totalRuns = runs * length(a3); % Total number of runs
N = nSym * osr; % Length of signal vector
h = rcosdesign(beta, span, osr, 'sqrt'); % Instantiate srrc signal
% Prepare for parallel execution on cluster
numA3 = length(a3);

parpool('local', 16);

% Pre-allocate block containers for results from each parallel worker
blockFeatL = cell(numA3, 1);
blockFeatNL = cell(numA3, 1);
blockL = cell(numA3, 1);
blockNL = cell(numA3, 1);

% Outer loop over nonlinear coefficients in parallel
parfor aIdx = 1:numA3
    a3Current = a3(aIdx);
    fprintf('Progress: a3 index %d/%d (a3=%.6f)\n', aIdx, numA3, a3Current);

    localFeatL = zeros(runs, 7);
    localFeatNL = zeros(runs, 7);
    localL = zeros(runs, N/2);
    localNL = zeros(runs, N/2);

    for r = 1:runs
        % Per-run accumulators
        featL = [];
        featNL = [];
        binsL = zeros(1, N);
        binsNL = zeros(1, N);

        % Simulation
        for i = 1:nFFT
            ak = 2 * randi([0 1], 1, nSym) - 1; % Randomly generate symbol vector of +1 and -1
            ak([1:5, end-4:end]) = 0; % Make first 5 and last 5 symbols 0
            ak = upsample(ak, osr); % Upsample symbol vector

            x = conv(ak, h, 'same'); % Make x
            xNL = a1.*x + a3Current.*(x.^3); % Make nonlinear x

            % Normalize xNL and x power
            normFactor = sqrt(mean(x.^2) + 1e-12);
            x = x / normFactor;
            xNL = xNL / normFactor;

            % Add noise
            Nvar = noiseRatio * sum(x.^2) / (length(x) - 1);
            noise = sqrt(Nvar) * randn(1, length(x)); % Make noise vector

            x = x + noise;
            xNL = xNL + noise;

            % Features dataset block: save only the first run
            if i == 1
                featL = x;
                featNL = xNL;
            end

            % Bins dataset block: average over nFFT runs
            % Make y (convolve at the receiver)
            yL = conv(x, h, 'same');
            yNL = conv(xNL, h, 'same');

            % Sum fft
            binsL = binsL + abs(fft(yL)).^2;
            binsNL = binsNL + abs(fft(yNL)).^2;
        end

        % Store per-run results into local matrices
        localFeatL(r, :) = extract_features(featL);
        localFeatNL(r, :) = extract_features(featNL);
        localL(r, :) = abs(binsL(1:N/2) / nFFT);
        localNL(r, :) = abs(binsNL(1:N/2) / nFFT);
    end

    blockFeatL{aIdx} = localFeatL;
    blockFeatNL{aIdx} = localFeatNL;
    blockL{aIdx} = localL;
    blockNL{aIdx} = localNL;
end

% Combine blocks from parallel runs into final datasets
dataL_feat = zeros(totalRuns, 7);
dataNL_feat = zeros(totalRuns, 7);
dataL_bins = zeros(totalRuns, N/2);
dataNL_bins = zeros(totalRuns, N/2);

for aIdx = 1:numA3
    rowRange = (aIdx - 1) * runs + (1:runs);
    dataL_feat(rowRange, :) = blockFeatL{aIdx};
    dataNL_feat(rowRange, :) = blockFeatNL{aIdx};
    dataL_bins(rowRange, :) = blockL{aIdx};
    dataNL_bins(rowRange, :) = blockNL{aIdx};
end

labels = [zeros(totalRuns, 1); ones(totalRuns, 1)];

% Write features CSV
featureNames = {
    'crest_factor', 'envelope_skewness', 'envelope_kurtosis', ...
    'peak_power', 'normalized_m4', 'spectral_flatness', 'spectral_entropy'
};
featDataset = array2table([[dataL_feat; dataNL_feat], labels], 'VariableNames', [featureNames, {'nonlinear'}]);
outputPath = '/work/pi_mduarte_umass_edu/oraftery_umass_edu/data/data.csv';
writetable(featDataset, outputPath);
fprintf('Saved data.csv features: %d rows x %d columns to %s\n', height(featDataset), width(featDataset), outputPath);

% Write bins CSV
binNames = arrayfun(@(k) sprintf('bin_%d', k), 0:(N/2)-1, 'UniformOutput', false);
binsDataset = array2table([[dataL_bins; dataNL_bins], labels], 'VariableNames', [binNames, {'nonlinear'}]);
outputPath = '/work/pi_mduarte_umass_edu/oraftery_umass_edu/data/bins_data.csv';
writetable(binsDataset, outputPath);
fprintf('Saved bins_data.csv: %d rows x %d columns to %s\n', height(binsDataset), width(binsDataset), outputPath);

% Engineered features function (supports averaging over nFFT signals if wanted)
function features = extract_features(signal)
    sig = signal(:);

    % Crest factor
    envelope = abs(sig);
    crest_factor = max(envelope) / (mean(envelope) + 1e-10);

    % Envelope skewness and kurtosis
    env_mean = mean(envelope);
    env_std = std(envelope);
    env_normalized = (envelope - env_mean) / (env_std + 1e-10);
    envelope_skewness = mean(env_normalized.^3);
    envelope_kurtosis = mean(env_normalized.^4);

    % Peak power
    peak_power = 10 * log10(max(abs(sig).^2) + 1e-10);

    % Normalized fourth moment
    m2 = mean(abs(sig).^2);
    m4 = mean(abs(sig).^4);
    normalized_m4 = m4 / (m2^2 + 1e-10);

    % Spectral flatness
    P = abs(fft(sig)).^2 / length(sig);
    P_mean = mean(P);
    P_normalized = P / (P_mean + 1e-10);
    spectral_flatness = exp(mean(log(P_normalized + 1e-10))) / (1 + 1e-10);

    % Spectral entropy
    P_sum = sum(P);
    P_pdf = P / (P_sum + 1e-10);
    spectral_entropy = -sum(P_pdf .* log2(P_pdf + 1e-10));

    features = [
        crest_factor, envelope_skewness, envelope_kurtosis, ...
        peak_power, normalized_m4, spectral_flatness, spectral_entropy
    ];
end