% parameters
nSym = 100; % number of symbols
nFFT = 100; % number of fft to average over
beta = 0.25; % rolloff factor 
span = 5; % number of symbols for srrc
osr = 16; % oversampling rate
a1 = 1; % linear coefficient
a3 = linspace(-0.05, -0.2, 10); % nonlinear cubic coefficient sweep
noiseRatio = .75; % what ratio of total linear power is used to determine noise variance
runs = 50; % number of independent runs for dataset generation
totalRuns = runs * length(a3);
N = nSym * osr; % length of signal vector
h = rcosdesign(beta, span, osr, 'sqrt'); % instantiate srrc signal

fprintf('Generating %d, continue?\n', length(a3) * runs * 2);
userInput = input("", "s");
if ~strcmpi(strtrim(userInput), 'y')
    return;
end

% pre-allocate result matrices
dataL  = zeros(totalRuns, N/2);
dataNL = zeros(totalRuns, N/2);

% outer loop over nonlinear coefficients, then runs
for aIdx = 1:length(a3)
    a3Current = a3(aIdx);
    for r = 1:runs
        rowIdx = (aIdx - 1) * runs + r;

        % instantiate per-run accumulators
        avgL  = zeros(1, N);
        avgNL = zeros(1, N);

        % simulation loop
        for i = 1:nFFT
            ak = 2 * randi([0 1], 1, nSym) - 1; % randomly generate symbol vector of +1 and -1
            ak([1:5, end-4:end]) = 0; % make first 5 and last 5 symbols 0
            ak = upsample(ak, osr); % upsample symbol vector

            x = conv(ak, h, 'same'); % make x
            xNL = a1.*x + a3Current.*(x.^3); % make nonlinear x

            % normalize xNL and x power
            normFactor = sqrt(mean(x.^2) + 1e-12);
            x = x / normFactor;
            xNL = xNL / normFactor;

            % Add the same noise realization to both signals
            Nvar = noiseRatio * sum(x.^2) / (length(x) - 1);
            noise = sqrt(Nvar) * randn(1, length(x));


            % Transmitted signals (what Willie observes)
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
        dataL(rowIdx, :) = abs(avgL(1:N/2) / nFFT);
        dataNL(rowIdx, :) = abs(avgNL(1:N/2) / nFFT);
    end
end

% build binary classification dataset (label 0 = linear (avgL),  label 1 = nonlinear (avgNL))
labels = [zeros(totalRuns, 1); ones(totalRuns, 1)];
allData = [dataL; dataNL];

% assemble table with descriptive column names
binNames = arrayfun(@(k) sprintf('bin_%d', k), 0:(N/2)-1, 'UniformOutput', false);
T = array2table([allData, labels], 'VariableNames', [binNames, {'nonlinear'}]);

% write to csv
outputPath = fullfile(fileparts(mfilename('fullpath')), '..', 'data', 'data.csv');
writetable(T, outputPath);
fprintf('Saved %d rows x %d columns to %s\n', height(T), width(T), outputPath);
