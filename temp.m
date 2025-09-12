% ============================
% Square-Root Raised Cosine (SRRC) Transmit and Receive Simulation
% ============================

% Parameters
Nsym = 1000;             % Number of data symbols
osr = 16;                % Oversampling rate (samples per symbol)
Ts = 1;                  % Symbol period
T = Ts/osr;              % Sample period
rolloff = 0.25;          % SRRC roll-off factor
span = 10;               % SRRC filter span in symbols

% --------------------------------------------------------
% Generate Random Data Symbols
% --------------------------------------------------------
a = randi([0 1], 1, Nsym)*2 - 1;   % Random +/-1 symbols
a(1:10) = 0;                       % First 10 set to zero
a(end-9:end) = 0;                  % Last 10 set to zero

% --------------------------------------------------------
% Design the SRRC Filter
% --------------------------------------------------------
p = rcosdesign(rolloff, span, osr, 'sqrt');   % SRRC filter

% --------------------------------------------------------
% Pulse Shaping (Transmitter)
% --------------------------------------------------------
a_up = upsample(a, osr);           % Insert zeros (length = 1000*16 = 16000)

% Convolve and trim to exactly 16000 samples
x_full = conv(a_up, p);            % Convolution (longer than 16000)
start_idx = floor(length(p)/2);    % Half filter delay
x = x_full(start_idx+1 : start_idx+16000);  % Extract central 16000 samples

% --------------------------------------------------------
% Matched Filtering (Receiver)
% --------------------------------------------------------
y_full = conv(x, p);               % Apply matched filter
y = y_full(start_idx+1 : start_idx+16000);  % Trim to 16000 samples

% ==== Prepare upsampled symbol impulses (length 16000) ====
samples_up = zeros(size(y));        % length(y) should be 16000
samples_up(1:osr:end) = samples;   % place the 1000 symbol samples at symbol instants
sym_pos = 1:osr:length(y);         % sample indices for symbol instants (1-based)

% Create a 16k vector with ±1 spikes at symbol instants
y_spikes = zeros(size(y));          % length 16000
y_spikes(1:osr:end) = samples;      % place the symbol values at correct positions

% Find nonzero indices
nz_idx = find(y_spikes ~= 0);

figure;

% (1) Tx waveform - full
subplot(2,2,1);
plot(1:length(x), x);
title('Transmitted Signal (Full)');
xlabel('Sample Index'); ylabel('Amplitude');

% (2) Tx waveform - zoomed
subplot(2,2,2);
plot(t_zoom(1):t_zoom(end), x(t_zoom(1):t_zoom(end)));
title('Transmitted Signal (Zoom: 20 Symbols)');
xlabel('Sample Index'); ylabel('Amplitude');

% (3) Matched filter output - full
subplot(2,2,3);
stem(nz_idx, y_spikes(nz_idx), 'filled');
title('Matched Filter Output (Full)');
xlabel('Sample Index'); ylabel('Amplitude');

% (4) Matched filter output - zoomed
subplot(2,2,4);
% keep only nonzeros in zoom range
zoom_idx = t_zoom(1):t_zoom(end);
nz_zoom = zoom_idx(y_spikes(zoom_idx) ~= 0);
stem(nz_zoom, y_spikes(nz_zoom), 'filled');
title('Matched Filter Output (Zoom: 20 Symbols)');
xlabel('Sample Index'); ylabel('Amplitude');
