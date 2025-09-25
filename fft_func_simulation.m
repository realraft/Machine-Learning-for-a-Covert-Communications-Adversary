%% Simulation

%% Parameters

Nsym = 1000; % number of symbols
Nfft = 100; % number of FFT to average
beta = 0.25; % rolloff factor
span = 10; % filter span in symbols
osr = 16; % oversampling rate
Ts = 1; % symbol period
avoid = 10; % number of samples at head and tail to avoid filter transients
a = 10; % amplitude of x
a1 = 1; % nonlinearity component from Jessica thesis 
a3 = -20; % nonlinearity component from Jessica thesis
var = 0.1; % noise variance

%% Calculations

T = Ts/osr; % sample period
fs = 1/T; % sample rate 
chunk = Nsym * osr; % chunk length

% create params structure for function
params = struct('Nsym', Nsym, 'Nfft', Nfft, 'osr', osr, 'avoid', avoid, 'a', a, 'a1', a1, 'a3', a3, 'var', var, 'chunk', chunk);

% p(t)
p = rcosdesign(beta, span, osr, 'sqrt'); % square root raised cosine

% FFT
y_nl = avgFFT(params, p, true, false); % take average fft (boolean 1 -> nonlinearities, boolean 2 -> noise)
y_l = avgFFT(params, p, false, false);

%% Plotting

f = (-chunk/2 : chunk/2-1) * (fs / chunk); % frequency vector from -fs/2 to fs/2
% compute PSDs (already returned as averaged power) and convert to dB
y_nl_sh = fftshift(y_nl); % center zero frequency for nonlinear case
y_l_sh  = fftshift(y_l);  % center zero frequency for linear case

y_nl_db = 10*log10(y_nl_sh + eps); % convert to dB, add eps to avoid log of zero
y_l_db  = 10*log10(y_l_sh + eps);

figure;
plot(f, y_l_db, '-b', 'LineWidth', 1.2); hold on; % linear (blue)
plot(f, y_nl_db, '--r', 'LineWidth', 1.2); % nonlinear (red dashed)
hold off;
xlabel('Frequency (Hz)');
ylabel('Power (dB)');
title('Average Power Spectral Density of y(t) (No Noise)');
legend('Linear','Nonlinear');
grid on;
xlim([-fs/2 fs/2]);

function y_fft =  avgFFT(params, p, nonlinearities, noise)
    % unpack needed parameters from struct
    Nsym = params.Nsym; 
    Nfft = params.Nfft; 
    osr = params.osr; 
    avoid = params.avoid; 
    a = params.a;
    a1 = params.a1; 
    a3 = params.a3; 
    var = params.var; 
    chunk = params.chunk;

    y_fft = zeros(1, chunk); % initialize average fft array
    for k = 1:Nfft

        % ak
        ak = 2*a*randi([0 1], 1, Nsym) - a; % randomly fill ak with +-a symbols
        ak = upsample(ak, osr); % upsample ak so it has size(chunk) samples
        ak(1:avoid) = 0; % zero specified number of samples at head
        ak(end-avoid+1:end) = 0; % zero specified number of samples at tail

        % x(t)
        x = conv(ak, p, 'same'); % perform summation
        if nonlinearities
            x = a1*x + a3*x.^3; % add nonlinearities
        end
        if noise
            x = x + sqrt(var) * randn(size(x)); % 0 mean gaussian, variance
        end

        % y(t)
        y = conv(x, p, 'same'); % x(t) * p(t)

        fft_chunk = fft(y, chunk); % take fft
        y_fft = y_fft + abs(fft_chunk).^2; % add all FFT up
    end
    y_fft = y_fft / Nfft; % take average
end