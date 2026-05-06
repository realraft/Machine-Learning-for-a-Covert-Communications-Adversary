% Parameters
nSym = 100; % Number of symbols
nFFT = 100; % Number of fft to average over
beta = 0.25; % Rolloff factor
span = 5; % Number of symbols for srrc
osr = 16; % Oversampling rate
a1 = 1; % Linear coefficient
a3 = linspace(-0.05, -0.2, 10); % Nonlinear cubic coefficient sweep
noiseRatio = 0.75; % What ratio of total linear power is used to determine noise variance
runs = 50; % Number of independent runs for dataset generation
totalRuns = runs * length(a3); % Total number of runs
N = nSym * osr; % Length of signal vector
h = rcosdesign(beta, span, osr, 'sqrt'); % Instantiate srrc signal

% Prompt user to continue
fprintf('Generating %d rows, continue? (y/n)\n', totalRuns*2);
userInput = input("", "s");
if ~strcmpi(strtrim(userInput), 'y')
    return;
end

% Pre allocate datasets
dataL_feat = zeros(totalRuns, 7);
dataNL_feat = zeros(totalRuns, 7);

dataL_bins = zeros(totalRuns, N/2);
dataNL_bins = zeros(totalRuns, N/2);

% Outer loop over nonlinear coefficients, inner for simulation of that a3
for aIdx = 1:length(a3)
    a3Current = a3(aIdx);
    fprintf('Progress: a3 index %d/%d (a3=%.6f)\n', aIdx, length(a3), a3Current);

    for r = 1:runs
        rowIdx = (aIdx - 1) * runs + r;

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

            % Normalize x
            x = x / sqrt(mean(abs(x).^2));

            % Apply nonlinearity
            xNL = a1.*x + a3Current.*(x.^3);
            
            % Normalize xNL
            xNL = xNL / sqrt(mean(abs(xNL).^2));

            % Now add independent noise
            noise_L  = sqrt(noiseRatio) * randn(1, length(x));
            noise_NL = sqrt(noiseRatio) * randn(1, length(x));

            x   = x + noise_L;
            xNL = xNL + noise_NL;

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

        % Append blocks to datasets (features from first iteration only)
        dataL_feat(rowIdx, :) = extract_features(featL);
        dataNL_feat(rowIdx, :) = extract_features(featNL);

        % Average fft and store magnitude (real-valued power) for each run (only right half of spectrum)
        dataL_bins(rowIdx, :) = abs(binsL(1:N/2) / nFFT);
        dataNL_bins(rowIdx, :) = abs(binsNL(1:N/2) / nFFT);
    end
end

labels = [zeros(totalRuns, 1); ones(totalRuns, 1)];

% Write features CSV
featureNames = {
    'crest_factor', 'envelope_skewness', 'envelope_kurtosis', ...
    'peak_power', 'normalized_m4', 'spectral_flatness', 'spectral_entropy'
};
featDataset = array2table([[dataL_feat; dataNL_feat], labels], 'VariableNames', [featureNames, {'nonlinear'}]);
outputPath = fullfile(fileparts(mfilename('fullpath')), '..', 'data', 'data.csv');
writetable(featDataset, outputPath);
fprintf('Saved data.csv features: %d rows x %d columns to %s\n', height(featDataset), width(featDataset), outputPath);

% Write bins CSV
binNames = arrayfun(@(k) sprintf('bin_%d', k), 0:(N/2)-1, 'UniformOutput', false);
binsDataset = array2table([[dataL_bins; dataNL_bins], labels], 'VariableNames', [binNames, {'nonlinear'}]);
outputPath = fullfile(fileparts(mfilename('fullpath')), '..', 'data', 'bins_data.csv');
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