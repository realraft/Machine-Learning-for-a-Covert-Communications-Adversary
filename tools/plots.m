clear; clc; close all;

nSym = 100;
nFFT = 256;
beta = 0.25;
span = 5;
osr = 16;
Rs = 1e6;
Fs = Rs * osr;
a1 = 1;
a3 = -0.18;
drive = 1.9;
noiseRatio = 0.05;

N = nSym * osr;
h = rcosdesign(beta, span, osr, 'sqrt');

avgL = zeros(1, N);
avgNL = zeros(1, N);

for i = 1:nFFT
	ak = 2 * randi([0 1], 1, nSym) - 1;
	ak([1:5, end-4:end]) = 0;
	akUp = upsample(ak, osr);

	x = conv(akUp, h, 'same');
	xDriven = drive .* x;
	xNL = a1 .* xDriven + a3 .* (xDriven .^ 3);

	noise = sqrt(noiseRatio) * randn(1, length(x));
	x = x + noise;
	xNL = xNL + noise;

	y = conv(x, h, 'same');
	yNL = conv(xNL, h, 'same');

	avgL = avgL + abs(fft(y)) .^ 2;
	avgNL = avgNL + abs(fft(yNL)) .^ 2;
end

psdL = fftshift(avgL / nFFT);
psdNL = fftshift(avgNL / nFFT);

psdL_dB = 10 * log10(psdL + eps);
psdNL_dB = 10 * log10(psdNL + eps);

smoothSpan = 17;
psdL_dB_s = movmean(psdL_dB, smoothSpan);
psdNL_dB_s = movmean(psdNL_dB, smoothSpan);

fHz = ((-N/2:N/2-1) / N) * Fs;

bandEdgeHz = (1 + beta) * Rs / 2;
inbandMask = abs(fHz) <= 0.8 * bandEdgeHz;
levelOffset = mean(psdL_dB_s(inbandMask)) - mean(psdNL_dB_s(inbandMask));
psdNL_dB_s_aligned = psdNL_dB_s + levelOffset;

zoomFactor = 4;
xZoomHz = zoomFactor * bandEdgeHz;
xMask = abs(fHz) <= xZoomHz;
yMin = min([psdL_dB_s(xMask), psdNL_dB_s_aligned(xMask)]) - 3;
yMax = max([psdL_dB_s(xMask), psdNL_dB_s_aligned(xMask)]) + 2;

figure('Color', 'w');
plot(fHz, psdL_dB_s, 'LineWidth', 1.8);
hold on;
plot(fHz, psdNL_dB_s_aligned, 'LineWidth', 1.8);
grid on;
xlabel('Frequency (Hz)');
ylabel('Power Spectral Density (dB)');
title('Linear vs Nonlinear Signal');
legend('Linear', 'Nonlinear (in-band aligned)', 'Location', 'southoutside', 'Orientation', 'horizontal');
xlim([-xZoomHz xZoomHz]);
ylim([yMin yMax]);

fprintf('Single-run spectral regrowth plot generated.\n');
fprintf('Used nFFT = %d averages and movmean span = %d to reduce bounce.\n', nFFT, smoothSpan);
fprintf('Regrowth controls: a3 = %.3f, drive = %.2f, noiseRatio = %.2f\n', a3, drive, noiseRatio);
fprintf('Rates: Rs = %.3e sym/s, Fs = %.3e samples/s\n', Rs, Fs);
fprintf('Displayed zoom = +/- %.3e Hz around DC, band edge = %.3e Hz\n', xZoomHz, bandEdgeHz);
