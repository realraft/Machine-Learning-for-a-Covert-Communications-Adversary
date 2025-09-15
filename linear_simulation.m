%% Simulation

%% Parameters

Nsym = 1000; % number of symbols
beta = 0.25; % rolloff factor
span = 10; % filter span in symbols
osr = 16; % oversampling rate
Ts = 1; % symbol period
T = Ts/osr; % sample period

% ak
ak = zeros(1, Nsym);
ak(11:Nsym-10) = 2*randi([0 1], 1, Nsym-20) - 1;  % randomly fill ak with -1 or +1
ak = upsample(ak, osr); % upsample ak so it has 16000 samples

%% Calculations

% p(t)
p = rcosdesign(beta, span, osr, 'sqrt'); % p(t), square root raised cosine

% x(t)
x = conv(ak, p, 'same'); % perform summation

% y(t)
y = conv(x, p, 'same'); % convolve to check for match filter

%% Plotting

figure(1);

% helpers
mid_range = @(N, hb, ha) max(1, floor(N/2)-hb) : min(N, floor(N/2)+ha);
sidx = @(N) 0:N-1;

% Figure 1 - Full width on top
subplot(4, 2, [1 2]);
L = length(p);
n_centered = -(L-1)/2 : (L-1)/2;
plot(n_centered, p);
xlabel('Sample Index');
ylabel('p(n)');
title('Square Root Raised Cosine Pulse p(n)');
grid on;
xlim([n_centered(1) n_centered(end)]);

% Figure 2 - Left side, second row
subplot(4, 2, 3);
plot(sidx(length(x)), x);
xlabel('Sample Index');
ylabel('x(t)');
title('Transmitted Waveform x(t)');
grid on;

% Figure 5 - Right side, second row
subplot(4, 2, 4);
x_zoom_idx = mid_range(length(x), 500, 499);
plot(x_zoom_idx-1, x(x_zoom_idx));
xlabel('Sample Index');
ylabel('x(t)');
title('Transmitted Waveform x(t) - Zoomed (1000 points from middle)');
grid on;

% Figure 3 - Left side, third row
subplot(4, 2, 5);
plot(sidx(length(y)), y);
xlabel('Sample Index');
ylabel('y(t)');
title('Matched Filter Output Waveform y(t)');
grid on;

% Figure 6 - Right side, third row
subplot(4, 2, 6);
y_zoom_idx = mid_range(length(y), 500, 499);
plot(y_zoom_idx-1, y(y_zoom_idx));
xlabel('Sample Index');
ylabel('y(t)');
title('Matched Filter Output Waveform y(t) - Zoomed (1000 points from middle)');
grid on;

% Figure 4 - Left side, fourth row
subplot(4, 2, 7);
d = (length(p)-1)/2;
idx = 1:osr:length(y);
idx = idx + d;
idx = idx(idx <= length(y));
peaks = sign(y(idx));
stem(idx, peaks, 'filled');
xlabel('Sample Index');
ylabel('y(t)');
title('Matched Filter Output Detected Symbol)');
grid on;

% Figure 7 - Right side, fourth row
subplot(4, 2, 8);
idx_zoom_range = mid_range(length(idx), 10, 9);
stem(idx(idx_zoom_range), peaks(idx_zoom_range), 'filled');
xlabel('Sample Index');
ylabel('y(t)');
title('Matched Filter Output Detected Symbol - Zoomed (20 points from middle)');
grid on;