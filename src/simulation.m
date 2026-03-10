% parameters
nSym = 100; % number of symbols
nFFT = 100; % number of fft to average over
beta = 0.25; % rolloff factor 
span = 5; % number of symbols for srrc
osr = 16; % oversampling rate
a1 = 1; % linear coefficient
a3 = -0.05; % nonlinear cubic coefficient
noiseRatio = .75; % what ratio of total linear power is used to determine noise variance
runs = 5000; % number of independent runs for dataset generation
N = nSym * osr; % length of signal vector

h = rcosdesign(beta, span, osr, 'sqrt'); % instantiate srrc signal

% pre-allocate result matrices (one row per run)
dataL  = zeros(runs, N/2);
dataNL = zeros(runs, N/2);

% find the theoretical band edge bin
bandwidth = (1 + beta) / 2;  % normalized bandwidth of SRRC
band_edge_bin = round(bandwidth * N / osr)

quit

% outer loop: each run produces one averaged spectrum sample
for r = 1:runs
    % instantiate per-run accumulators
    avgL  = zeros(1, N);
    avgNL = zeros(1, N);

    % simulation loop
    for i = 1:nFFT
        ak = 2 * randi([0 1], 1, nSym) - 1; % randomly generate symbol vector of +1 and -1
        ak([1:5, end-4:end]) = 0; % make first 5 and last 5 symbols 0
        ak = upsample(ak, osr); % upsample symbol vector

        x = conv(ak, h, 'same'); % make x
        xNL = a1.*x + a3.*(x.^3); % make nonlinear x

        % calculate noise variance
        noiseVar = noiseRatio * mean(x.^2); % variance is 75% of total linear power

        % normalize xNL and x power
        powerRatio = mean(xNL.^2) / mean(x.^2); % compute power ratio for normalization
        xNL = xNL / sqrt(powerRatio);

        % add noise
        noise = sqrt(noiseVar) * randn(1, length(x)); % make noise vector
        x = x + noise;
        xNL = xNL + noise;

        % make y (convolve at the receiver)
        y = conv(x, h, 'same');
        yNL = conv(xNL, h, 'same');

        % sum fft
        avgL  = avgL  + abs(fft(y)).^2;
        avgNL = avgNL + abs(fft(yNL)).^2;
    end

    % average fft and store magnitude (real-valued power) for each run (only right half of spectrum)
    dataL(r, :) = abs(avgL(1:N/2) / nFFT);
    dataNL(r, :) = abs(avgNL(1:N/2) / nFFT);
end

% build binary classification dataset (label 0 = linear (avgL),  label 1 = nonlinear (avgNL))
labels = [zeros(runs, 1); ones(runs, 1)];
allData = [dataL; dataNL];

% assemble table with descriptive column names
binNames = arrayfun(@(k) sprintf('bin_%d', k), 0:(N/2)-1, 'UniformOutput', false);
T = array2table([allData, labels], 'VariableNames', [binNames, {'nonlinear'}]);

% write to csv
outputPath = fullfile(fileparts(mfilename('fullpath')), '..', 'data', 'data.csv');
writetable(T, outputPath);
fprintf('Saved %d rows x %d columns to %s\n', height(T), width(T), outputPath);
