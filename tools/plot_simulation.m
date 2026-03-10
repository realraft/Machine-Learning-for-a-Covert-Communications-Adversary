% parameters
nSym = 100; % number of symbols
nFFT = 100; % number of fft to average over
beta = 0.25; % rolloff factor 
span = 5; % number of symbols for srrc
osr = 16; % oversampling rate
a1 = 1; % linear coefficient
a3 = -0.05; % nonlinear cubic coefficient
noiseRatio = .75; % what ratio of total linear power is used to determine noise variance
N = nSym * osr; % length of signal vector

h = rcosdesign(beta, span, osr, 'sqrt'); % instantiate srrc signal

% single-run accumulators
avgL  = zeros(1, N);
avgNL = zeros(1, N);

% simulation loop (single run, averaged over nFFT)
for i = 1:nFFT
    ak = 2 * randi([0 1], 1, nSym) - 1;
    ak([1:5, end-4:end]) = 0;
    ak = upsample(ak, osr);

    x = conv(ak, h, 'same');
    xNL = a1.*x + a3.*(x.^3);

    noiseVar = noiseRatio * mean(x.^2);

    powerRatio = mean(xNL.^2) / mean(x.^2);
    xNL = xNL / sqrt(powerRatio);

    noise = sqrt(noiseVar) * randn(1, length(x));
    x = x + noise;
    xNL = xNL + noise;

    y = conv(x, h, 'same');
    yNL = conv(xNL, h, 'same');

    avgL  = avgL  + abs(fft(y)).^2;
    avgNL = avgNL + abs(fft(yNL)).^2;
end

% average and convert to dB
specL  = 10*log10(avgL  / nFFT);
specNL = 10*log10(avgNL / nFFT);

% frequency axis (normalized, centered)
f = (-N/2 : N/2-1) / N;

% fftshift for centered plot
specL_shift  = fftshift(specL);
specNL_shift = fftshift(specNL);

% plot
figure('Color', 'w', 'Position', [100 100 900 500]);
plot(f, specL_shift,  'b',   'LineWidth', 1.5, 'DisplayName', 'Linear');
hold on;
plot(f, specNL_shift, 'r--', 'LineWidth', 1.5, 'DisplayName', 'Nonlinear');
hold off;

xlabel('Normalized Frequency (cycles/sample)');
ylabel('Power Spectral Density (dB)');
title(sprintf('Averaged Power Spectrum (nFFT=%d)  —  Linear vs Nonlinear', nFFT));
legend('Location', 'best');
grid on;
xlim([-0.5 0.5]);