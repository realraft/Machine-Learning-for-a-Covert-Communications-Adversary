%% Simulation

%% Parameters

Nsym = 1000; % number of symbols
Nfft = 100; % number of FFT to average
beta = 0.25; % rolloff factor
span = 10; % filter span in symbols
osr = 16; % oversampling rate
Ts = 1; % symbol period
avoid = 10; % number of samples at head and tail to avoid filter transients
a1 = 1; % nonlinearity component from Jessica thesis 
a3 = -0.05; % nonlinearity component from Jessica thesis
var = 1; % noise variance

%% Calculations

T = Ts/osr; % sample period
fs = 1/T; % sample rate 
chunk = Nsym * osr; % chunk length

% p(t)
p = rcosdesign(beta, span, osr, 'sqrt'); % square root raised cosine

% FFT
y_fft = zeros(1, chunk); % initialize average fft array
for k = 1:Nfft

    % ak
    ak = 2*randi([0 1], 1, Nsym) - 1; % randomly fill ak with +-1 symbols
    ak = upsample(ak, osr); % upsample ak so it has size(chunk) samples
    ak(1:avoid) = 0; % zero specified number of samples at head
    ak(end-avoid+1:end) = 0; % zero specified number of samples at tail

    % x(t)
    x = conv(ak, p, 'same'); % perform summation
    x = a1*x + a3*x.^3; % add nonlinearities
    x = x + sqrt(var) * randn(size(x)); % 0 mean gaussian, variance

    % y(t)
    y = conv(x, p, 'same'); % x(t) * p(t)

    fft_chunk = fft(y, chunk); % take fft
    y_fft = y_fft + abs(fft_chunk).^2; % add all FFT up
end
y_fft = y_fft / Nfft; % take average

%% Plotting
f = (-chunk/2 : chunk/2-1) * (fs / chunk); % frequency vector from -fs/2 to fs/2
y_fft_sh = fftshift(y_fft); % center zero frequency
y_fft_db = 10*log10(y_fft_sh + eps); % convert to dB, add eps to avoid log of zero

figure;
plot(f, y_fft_db);
xlabel('Frequency (Hz)');
ylabel('Power (dB)');
title('Average Power Spectral Density of y(t)');
grid on;
xlim([-fs/2 fs/2]);
