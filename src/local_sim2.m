% Parameters
nSym = 100; % Number of symbols
beta = 0.25; % Rolloff factor
span = 5; % Number of symbols for srrc
osr = 16; % Oversampling rate
a1 = 1; % Linear coefficient
a3 = linspace(-0.2, -0.5, 10); % Nonlinear cubic coefficient sweep
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
dataL_raw = zeros(totalRuns, 2*N);
dataNL_raw = zeros(totalRuns, 2*N);

% Outer loop over nonlinear coefficients, inner for simulation of that a3
for aIdx = 1:length(a3)
    a3Current = a3(aIdx);
    fprintf('Progress: a3 index %d/%d (a3=%.6f)\n', aIdx, length(a3), a3Current);

    for r = 1:runs
        rowIdx = (aIdx - 1) * runs + r;

        % Simulation
        % Generate complex baseband signal
        ak = (2*randi([0 1], 1, nSym) - 1) + 1j*(2*randi([0 1], 1, nSym) - 1);
        ak = ak / sqrt(2); % Normalize QPSK power
        ak([1:5, end-4:end]) = 0; % Make first 5 and last 5 symbols 0
        ak = upsample(ak, osr); % Upsample symbol vector

        x = conv(ak, h, 'same'); % Make x

        % Normalize x
        x = x / sqrt(mean(abs(x).^2));

        % Apply nonlinearity
        xNL = a1.*x + a3Current.*(x.^3);
        
        % Normalize xNL
        xNL = xNL / sqrt(mean(abs(xNL).^2));

        % Add noise
        noise_L  = sqrt(noiseRatio) * (randn(1, length(x)) + 1j*randn(1, length(x))) / sqrt(2);
        noise_NL = sqrt(noiseRatio) * (randn(1, length(x)) + 1j*randn(1, length(x))) / sqrt(2);
        x   = x   + noise_L;
        xNL = xNL + noise_NL;

        % Save raw IQ samples for this run [Re, Im]
        dataL_raw(rowIdx, :) = [real(x), imag(x)];
        dataNL_raw(rowIdx, :) = [real(xNL), imag(xNL)];
    end
end

% --- TEMPORARY SANITY CHECK ---
% Sanity check: are the classes actually different?
sampleDiff = mean(abs(dataNL_raw(1:10,:) - dataL_raw(1:10,:)).^2, 2);
fprintf('Mean squared difference between NL and L (first 10 rows): %.6f\n', mean(sampleDiff));
if mean(sampleDiff) < 1e-6
    warning('Classes may be too similar to distinguish!');
end
% ------------------------------

labels = [zeros(totalRuns, 1); ones(totalRuns, 1)];

% Write raw IQ CSV
reNames = arrayfun(@(k) sprintf('re_%d', k), 1:N, 'UniformOutput', false);
imNames = arrayfun(@(k) sprintf('im_%d', k), 1:N, 'UniformOutput', false);
allNames = [reNames, imNames];

outputPath = fullfile(fileparts(mfilename('fullpath')), '..', 'data', 'data.csv');

delete(outputPath)

% Write headers manually for performance
fid = fopen(outputPath, 'w');
fprintf(fid, '%s,', allNames{:});
fprintf(fid, 'nonlinear\n');
fclose(fid);

% Write data matrix using writematrix
data = [[dataL_raw; dataNL_raw], labels];
writematrix(data, outputPath, 'WriteMode', 'append');

fprintf('Saved data.csv raw IQ samples: %d rows x %d columns to %s\n', size(data, 1), size(data, 2), outputPath);