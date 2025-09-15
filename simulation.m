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

%% Calculations

T = Ts/osr; % sample period
fs = 1/T; % sample rate 
chunk = Nsym * osr; % chunk length
valid_idx = (avoid+1) : (chunk-avoid); % valid indexes to insert symbols

% p(t)
p = rcosdesign(beta, span, osr, 'sqrt'); % square root raised cosine

% FFT
y_fft = zeros(1, chunk); % initialize average fft array
for k = 1:Nfft

    % ak
    ak = zeros(1, chunk); % initialize ak
    sym_pos = valid_idx(randperm(numel(valid_idx), Nsym)); % randomly choose symbol positions
    sym = 2*randi([0 1], 1, Nsym) - 1; % randomly choose symbol sign
    ak(sym_pos) = sym; % insert symbols

    % x(t)
    x = conv(ak, p, 'same'); % perform summation
    x = a1*x + a3*x.^3; % add nonlinearities

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
