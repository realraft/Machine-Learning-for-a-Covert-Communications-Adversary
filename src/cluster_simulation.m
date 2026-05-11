%% Parameters
nSym = 100; % Number of symbols
nFFT = 100; % Number of FFT realizations to average for PSD
beta = 0.25; % SRRC roll-off
span = 5; % SRRC span (symbols)
osr = 16; % Oversampling rate
a1 = 1; % Linear coefficient
a3_values = [-0.08, -0.16, -0.24, -0.32, -0.45, -0.60, -0.80]; % Nonlinearity sweep
runs = 15000; % Independent runs per a3 value
numWorkers = 16; % Parallel workers

N = nSym * osr; % Signal length (samples per realization)
h = rcosdesign(beta, span, osr, 'sqrt'); % SRRC pulse
numA3 = length(a3_values);
totalRuns = numA3 * runs;

%% Cluster I/O path
dataDir = '/work/pi_mduarte_umass_edu/oraftery_umass_edu/data';
if ~exist(dataDir, 'dir'); mkdir(dataDir); end
outputPath = fullfile(dataDir, 'simulation_data.h5');

%% Parallel pool
if isempty(gcp('nocreate'))
    parpool('local', numWorkers);
end

%% Pre-allocate output arrays
% Naming convention: H0 = Alice absent (label 0), H1 = Alice present (label 1)
allIQH0   = zeros(totalRuns, 2*N);
allIQH1   = zeros(totalRuns, 2*N);
allBinsH0 = zeros(totalRuns, N);
allBinsH1 = zeros(totalRuns, N);
allFeatH0 = zeros(totalRuns, 7);
allFeatH1 = zeros(totalRuns, 7);
allA3 = zeros(totalRuns, 1);

%% Parallelized loop
parfor k = 1:totalRuns
    aIdx = ceil(k / runs);
    runIdx = k - (aIdx - 1) * runs;
    a3Current = a3_values(aIdx);
    if runIdx == 1
        fprintf('Starting a3=%.4f (index %d/%d)\n', a3Current, aIdx, numA3);
    end

    % Run accumulators
    accBinsH0 = zeros(1, N);
    accBinsH1 = zeros(1, N);
    iqH0_row  = zeros(1, 2*N);
    iqH1_row  = zeros(1, 2*N);
    featH0_row = zeros(1, 7);
    featH1_row = zeros(1, 7);

    for i = 1:nFFT
        % Alice's Signal
        ak_alice = (randn(1, nSym) + 1j*randn(1, nSym)) / sqrt(2);
        ak_alice([1:5, end-4:end]) = 0;
        ak_alice = upsample(ak_alice, osr);
        alice = conv(ak_alice, h, 'same');
        alice_NL = a1.*alice + a3Current.*(alice.^3);

        % Jammer's signal
        ak_jam = (randn(1, nSym) + 1j*randn(1, nSym)) / sqrt(2);
        ak_jam([1:5, end-4:end]) = 0;
        ak_jam = upsample(ak_jam, osr);
        jammer = conv(ak_jam, h, 'same');

        % Hypotheses
        % H0 (Alice absent):  signal = jammer
        % H1 (Alice present): signal = alice_NL + jammer
        sig_H0 = jammer;
        sig_H1 = alice_NL + jammer;

        % Normalize signals to unit power
        sig_H0 = sig_H0 / sqrt(mean(abs(sig_H0).^2) + 1e-12);
        sig_H1 = sig_H1 / sqrt(mean(abs(sig_H1).^2) + 1e-12);

        % Independent noise for each hypothesis
        noiseH0 = sqrt(noiseRatio/2) * (randn(1, N) + 1j*randn(1, N));
        noiseH1 = sqrt(noiseRatio/2) * (randn(1, N) + 1j*randn(1, N));
        rH0 = sig_H0 + noiseH0;
        rH1 = sig_H1 + noiseH1;

        % First realization defines the raw I/Q row and feature row
        if i == 1
            iqH0_row   = [real(rH0), imag(rH0)];
            iqH1_row   = [real(rH1), imag(rH1)];
            featH0_row = extract_features(rH0);
            featH1_row = extract_features(rH1);
        end

        % Averaged PSD estimate
        accBinsH0 = accBinsH0 + abs(fft(rH0)).^2;
        accBinsH1 = accBinsH1 + abs(fft(rH1)).^2;
    end

    allIQH0(k, :)   = iqH0_row;
    allIQH1(k, :)   = iqH1_row;
    allFeatH0(k, :) = featH0_row;
    allFeatH1(k, :) = featH1_row;
    allBinsH0(k, :) = accBinsH0 / nFFT;
    allBinsH1(k, :) = accBinsH1 / nFFT;
    allA3(k) = a3Current;
end

%% Stack H0 (label 0) and H1 (label 1)
labels = uint8([zeros(totalRuns, 1); ones(totalRuns, 1)]);
a3Col  = [allA3; allA3];

iqAll   = [allIQH0;   allIQH1];
binsAll = [allBinsH0; allBinsH1];
featAll = [allFeatH0; allFeatH1];

featureNames = {'crest_factor','envelope_skewness','envelope_kurtosis', ...
                'peak_power','normalized_m4','spectral_flatness','spectral_entropy'};

%% Write HDF5 (one file, three groups: /iq, /bins, /feat)
if exist(outputPath, 'file'); delete(outputPath); end

write_group(outputPath, '/iq',   iqAll,   a3Col, labels);
write_group(outputPath, '/bins', binsAll, a3Col, labels);
write_group(outputPath, '/feat', featAll, a3Col, labels);

h5writeatt(outputPath, '/', 'a3_values',   a3_values);
h5writeatt(outputPath, '/', 'noiseRatio',  noiseRatio);
h5writeatt(outputPath, '/', 'N',           int32(N));
h5writeatt(outputPath, '/', 'nFFT',        int32(nFFT));
h5writeatt(outputPath, '/', 'runs_per_a3', int32(runs));

fprintf('Wrote %s\n', outputPath);
fprintf('  /iq/data    : %d x %d\n', size(iqAll, 1),   size(iqAll, 2));
fprintf('  /bins/data  : %d x %d\n', size(binsAll, 1), size(binsAll, 2));
fprintf('  /feat/data  : %d x %d\n', size(featAll, 1), size(featAll, 2));


%% Helpers

function write_group(path, group, data, a3Col, labels)
    h5create(path, [group '/data'],      size(data),     'Datatype', 'double');
    h5create(path, [group '/a3'],        size(a3Col),    'Datatype', 'double');
    h5create(path, [group '/nonlinear'], size(labels),   'Datatype', 'uint8');
    h5write(path, [group '/data'],      data);
    h5write(path, [group '/a3'],        a3Col);
    h5write(path, [group '/nonlinear'], labels);
end

function features = extract_features(signal)
    sig = signal(:);
    envelope = abs(sig);

    crest_factor = max(envelope) / (mean(envelope) + 1e-10);

    env_mean = mean(envelope);
    env_std  = std(envelope);
    env_norm = (envelope - env_mean) / (env_std + 1e-10);
    envelope_skewness = mean(env_norm.^3);
    envelope_kurtosis = mean(env_norm.^4);

    peak_power = 10 * log10(max(abs(sig).^2) + 1e-10);

    m2 = mean(abs(sig).^2);
    m4 = mean(abs(sig).^4);
    normalized_m4 = m4 / (m2^2 + 1e-10);

    P = abs(fft(sig)).^2 / length(sig);
    P_mean = mean(P);
    P_norm = P / (P_mean + 1e-10);
    spectral_flatness = exp(mean(log(P_norm + 1e-10))) / (1 + 1e-10);

    P_pdf = P / (sum(P) + 1e-10);
    spectral_entropy = -sum(P_pdf .* log2(P_pdf + 1e-10));

    features = [crest_factor, envelope_skewness, envelope_kurtosis, ...
                peak_power, normalized_m4, spectral_flatness, spectral_entropy];
end
