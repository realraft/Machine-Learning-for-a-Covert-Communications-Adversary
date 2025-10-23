%% Plot linear and nonlinear signals along with FFT power in dB.

%% Setup helper path
scriptDir = fileparts(mfilename('fullpath'));
helperDir = fullfile(scriptDir, '..', 'data', 'helper');
if exist(helperDir, 'dir')
    addpath(helperDir);
end

%% Editable parameters
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
noiseVar = 0.1;
includeNonlinearity = true;
includeNoise = true;

%% Simulation setup
params = struct('Nsym', Nsym, ...
                'Nfft', Nfft, ...
                'osr', osr, ...
                'avoid', avoid, ...
                'a', a, ...
                'a1', a1, ...
                'a3', a3, ...
                'noiseVar', noiseVar);

pulse = rcosdesign(beta, span, osr, 'sqrt');

[linearPower, yLinear] = compute_fft_average(params, pulse, false, includeNoise);
[nonlinearPower, yNonlinear] = compute_fft_average(params, pulse, true, includeNoise);

if ~includeNonlinearity
    yNonlinear = yLinear;
    nonlinearPower = linearPower;
end

%% Frequency grid
sampleInterval = Ts / osr;
sampleRate = 1 / sampleInterval;
numSamples = Nsym * osr;
frequency = (-numSamples/2 : numSamples/2-1) * (sampleRate / numSamples);
linearPSD = fftshift(linearPower);
nonlinearPSD = fftshift(nonlinearPower);
linearPSDdB = 10 * log10(linearPSD + eps);
nonlinearPSDdB = 10 * log10(nonlinearPSD + eps);
time = (0:numSamples-1) * sampleInterval;

%% Plots
figure('Name', 'Signals and FFT Magnitudes', 'Color', 'w');

subplot(2, 1, 1);
plot(time, yLinear, 'b', 'LineWidth', 1.1);
hold on;
plot(time, yNonlinear, '--r', 'LineWidth', 1.1);
hold off;
xlabel('Time (s)');
ylabel('Amplitude');
title('Matched Filter Output');
legend('Linear', 'Nonlinear', 'Location', 'best');
grid on;
xlim([time(1) time(end)]);

subplot(2, 1, 2);
plot(frequency, linearPSDdB, 'b', 'LineWidth', 1.1);
hold on;
plot(frequency, nonlinearPSDdB, '--r', 'LineWidth', 1.1);
hold off;
xlabel('Frequency (Hz)');
ylabel('Power (dB)');
title('Average Power Spectral Density');
legend('Linear', 'Nonlinear', 'Location', 'best');
grid on;
xlim([frequency(1) frequency(end)]);
