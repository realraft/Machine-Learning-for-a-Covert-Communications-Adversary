%% Plot linear and nonlinear signals along with FFT power in dB.

% Editable parameters -----------------------------------------------------
scriptDir = fileparts(mfilename('fullpath'));
addpath(scriptDir);

Nsym = 1000;
Nfft = 100;
beta = 0.25;
span = 10;
osr = 16;
Ts = 1;
avoid = 10;
a = 1.4678;
a1 = 1;
a3 = -2.5261;
noiseVariance = 0.1;
includeNonlinearity = true;
includeNoise = false;

% Simulation setup --------------------------------------------------------
params = struct('Nsym', Nsym, ...
                'Nfft', Nfft, ...
                'osr', osr, ...
                'avoid', avoid, ...
                'a', a, ...
                'a1', a1, ...
                'a3', a3, ...
                'noiseVar', noiseVariance);

pulse = rcosdesign(beta, span, osr, 'sqrt');

[linearPower, yLinear] = compute_fft_average(params, pulse, false, includeNoise);
[nonlinearPower, yNonlinear] = compute_fft_average(params, pulse, true, includeNoise);

if ~includeNonlinearity
    yNonlinear = yLinear;
    nonlinearPower = linearPower;
end

% Frequency grid ----------------------------------------------------------
T = Ts / osr;
fs = 1 / T;
chunk = Nsym * osr;
fVector = (-chunk/2 : chunk/2-1) * (fs / chunk);
linearPSD = fftshift(linearPower);
nonlinearPSD = fftshift(nonlinearPower);

linearPSD_dB = 10 * log10(linearPSD + eps);
nonlinearPSD_dB = 10 * log10(nonlinearPSD + eps);

timeAxis = (0:chunk-1) * T;

% Plots -------------------------------------------------------------------
figure('Name', 'Signals and FFT Magnitudes', 'Color', 'w');

subplot(2,1,1);
plot(timeAxis, yLinear, 'b', 'LineWidth', 1.1); hold on;
plot(timeAxis, yNonlinear, '--r', 'LineWidth', 1.1);
hold off;
xlabel('Time (s)');
ylabel('Amplitude');
title('Matched Filter Output');
legend('Linear', 'Nonlinear', 'Location', 'best');
grid on;
xlim([timeAxis(1) timeAxis(end)]);

subplot(2,1,2);
plot(fVector, linearPSD_dB, 'b', 'LineWidth', 1.1); hold on;
plot(fVector, nonlinearPSD_dB, '--r', 'LineWidth', 1.1);
hold off;
xlabel('Frequency (Hz)');
ylabel('Power (dB)');
title('Average Power Spectral Density');
legend('Linear', 'Nonlinear', 'Location', 'best');
grid on;
xlim([fVector(1) fVector(end)]);
