% parameters
nSym = 512; % number of symbols
nFFT = 100; % number of fft to average over
beta = 0.25; % rolloff factor 
span = 10; % number of symbols for srrc (!!!ASK GOECKEL CORRECT VALUE HERE!!!)
osr = 16; % oversampling rate
a1 = 1; % linear coefficient
a3 = -0.25; % nonlinear cubic coefficient
noiseVar = 50; % what ratio of total power is used to determine noise variance
N = nSym * osr; % length of signal vector

h = rcosdesign(beta, span, osr, 'sqrt'); % instantiate srrc signal

% instantiate final signal vector
avgL = zeros(1, N);
avgNL  = zeros(1, N);

% simulation loop
for i = 1:nFFT
    ak = 2 * randi([0 1], 1, nSym) - 1; % randomly generate symbol vector of +1 and -1
    ak = upsample(ak, osr); % upsample symbol vector

    x = conv(ak, h, 'same'); % make x
    xNL = a1.*x + a3.*(x.^3); % make nonlinear x

    % add noise
    noise = sqrt(noiseVar) * randn(1, length(x)); % make noise vector
    x = x + noise;
    xNL = xNL + noise;

    % normalize xNL and x power
    powerRatio = mean(xNL.^2) / mean(x.^2); % compute power ratio for normalization
    xNL = xNL / sqrt(powerRatio);

    % make y
    y = conv(x, h, 'same');
    yNL = conv(xNL, h, 'same');

    % sum fft
    avgL = avgL + abs(fft(y)).^2;
    avgNL  = avgNL  + abs(fft(yNL)).^2;
end

% average fft
avgL = avgL / nFFT;
avgNL = avgNL / nFFT;

% plot averaged FFTs — only first half of spectrum
halfN = floor(N/2) + 1;
freqAxis = (0:halfN-1) / N; % normalized frequency axis (0 to 0.5)

figure;
plot(freqAxis, 10*log10(avgL(1:halfN)), 'b', 'DisplayName', 'Linear');
hold on;
plot(freqAxis, 10*log10(avgNL(1:halfN)), 'r--', 'DisplayName', 'Nonlinear');
xlabel('Normalized Frequency');
ylabel('Power (dB)');
title('Averaged FFT of Linear vs Nonlinear Signals');
legend show;
grid on;