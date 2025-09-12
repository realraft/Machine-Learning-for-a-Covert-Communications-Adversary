% Parameters
Nsym = 1000;             % Number of data symbols to transmit
osr = 16;                % Oversampling rate (samples per symbol)
Ts = 1;                  % Symbol period (arbitrary = 1 unit time)
T = Ts/osr;              % Sampling period (time per sample)
rolloff = 0.25;          % Roll-off factor for SRRC filter (controls excess bandwidth)
span = 10;               % SRRC filter span in symbols (filter length = span*osr + 1 samples)

% --------------------------------------------------------
% Generate Random Data Symbols
% --------------------------------------------------------
% Generate +/-1 values randomly to simulate BPSK symbols.
a = randi([0 1], 1, Nsym)*2 - 1;

% Force the first and last 10 symbols to be zero.
% This is common in comms simulations to avoid startup transients
% where the filter ramp-up and ramp-down would distort edges.
a(1:10) = 0;
a(end-9:end) = 0;

% --------------------------------------------------------
% Design the SRRC Filter
% --------------------------------------------------------
% rcosdesign generates a raised cosine or root-raised cosine filter.
% Inputs:
%   rolloff = excess bandwidth factor
%   span    = filter length in symbols
%   osr     = oversampling rate
%   'sqrt'  = square-root raised cosine (SRRC)
% This filter will both shape the transmit pulse and act as the matched filter.
p = rcosdesign(rolloff, span, osr, 'sqrt');

% --------------------------------------------------------
% Pulse Shaping (Transmitter)
% --------------------------------------------------------
% To apply pulse shaping:
%   1. Insert osr-1 zeros between each symbol ("upsampling").
%   2. Convolve with SRRC filter to smooth transitions and limit bandwidth.
a_up = upsample(a, osr);           % Insert zeros between symbols
x = conv(a_up, p, 'same');         % Convolve with SRRC filter (Tx waveform)
                                   % 'same' keeps the output length aligned with a_up

% --------------------------------------------------------
% Matched Filtering (Receiver)
% --------------------------------------------------------
% At the receiver, apply another SRRC filter.
% The cascade of Tx SRRC + Rx SRRC is equivalent to a raised cosine filter,
% which has the Nyquist property (zero ISI at symbol instants).
y = conv(x, p, 'same');            % Filter the received signal

% --------------------------------------------------------
% Symbol Synchronization
% --------------------------------------------------------
% The matched filter output contains one strong peak per symbol period.
% Since we oversampled by osr, we pick one sample every osr samples.
samples = y(1:osr:end);            % Extract symbol-spaced samples

% --------------------------------------------------------
% Prepare Indices for Zoomed Plots
% --------------------------------------------------------
% Pick 20 symbols around the middle of the frame for detailed view.
mid = floor(Nsym/2);
rng = (mid-10):(mid+9);            % 20 consecutive symbols
t_zoom = rng*osr;                  % Corresponding sample indices

% Create one figure with 4 subplots
figure;

% (1) Tx waveform - full
subplot(2,2,1);
plot(x);
title('Transmitted Signal (Full)');
xlabel('Sample Index'); ylabel('Amplitude');

% (2) Tx waveform - zoomed
subplot(2,2,2);
plot(x(t_zoom(1):t_zoom(end)));
title('Transmitted Signal (Zoom: 20 Symbols)');
xlabel('Sample Index'); ylabel('Amplitude');

% (3) Matched filter samples - full
subplot(2,2,3);
stem(samples);
title('Matched Filter Output (Full)');
xlabel('Symbol Index'); ylabel('Amplitude');

% (4) Matched filter samples - zoomed
subplot(2,2,4);
stem(rng, samples(rng));
title('Matched Filter Output (Zoom: 20 Symbols)');
xlabel('Symbol Index'); ylabel('Amplitude');
