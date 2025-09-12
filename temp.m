%% Square-Root Raised Cosine (SRRC) Transmit and Receive Simulation
% Complete communication system simulation demonstrating SRRC pulse shaping
% at the transmitter and matched filtering at the receiver.
% This implementation shows the full transmit-receive chain with proper
% signal processing techniques.

% Clear workspace and close figures
clc;
clear;
close all;

%% System Parameters
NUM_DATA_SYMBOLS = 1000;            % Number of data symbols to transmit
OVERSAMPLING_RATIO = 16;            % Samples per symbol (oversampling factor)
SYMBOL_PERIOD = 1;                  % Symbol period (Ts)
SAMPLE_PERIOD = SYMBOL_PERIOD / OVERSAMPLING_RATIO;  % Sample period (T)
ROLLOFF_FACTOR = 0.25;              % SRRC roll-off factor (beta)
FILTER_SPAN_SYMBOLS = 10;           % SRRC filter span in symbol periods
NUM_ZERO_PADDING = 10;              % Zero symbols at start and end

%% Generate Random Data Symbols
% Create random binary symbols (+1/-1 for BPSK)
data_symbols = randi([0, 1], 1, NUM_DATA_SYMBOLS) * 2 - 1;

% Add zero padding to minimize edge effects
data_symbols(1:NUM_ZERO_PADDING) = 0;
data_symbols(end-NUM_ZERO_PADDING+1:end) = 0;

%% Design Square-Root Raised Cosine Filter
% Create SRRC pulse shaping filter (used at both TX and RX)
srrc_filter = rcosdesign(ROLLOFF_FACTOR, FILTER_SPAN_SYMBOLS, ...
                        OVERSAMPLING_RATIO, 'sqrt');

%% Transmitter: Pulse Shaping
% Upsample data symbols by inserting zeros
upsampled_symbols = upsample(data_symbols, OVERSAMPLING_RATIO);

% Apply pulse shaping filter (convolution)
transmitted_signal_full = conv(upsampled_symbols, srrc_filter);

% Calculate filter delay and extract central portion
filter_delay_samples = floor(length(srrc_filter) / 2);
target_length = NUM_DATA_SYMBOLS * OVERSAMPLING_RATIO;
start_index = filter_delay_samples + 1;
end_index = start_index + target_length - 1;

% Extract transmitted signal (compensating for filter delay)
transmitted_signal = transmitted_signal_full(start_index:end_index);

%% Receiver: Matched Filtering
% Apply matched filter (same SRRC filter used at transmitter)
received_signal_full = conv(transmitted_signal, srrc_filter);

% Extract received signal (compensating for filter delay)
received_signal = received_signal_full(start_index:end_index);

%% Sample at Symbol Instants
% Extract samples at symbol timing instants
symbol_sample_indices = 1:OVERSAMPLING_RATIO:length(received_signal);
recovered_symbols = received_signal(symbol_sample_indices);

% Create impulse representation for visualization
symbol_impulses = zeros(size(received_signal));
symbol_impulses(symbol_sample_indices) = recovered_symbols;

% Find non-zero symbol positions for plotting
nonzero_indices = find(symbol_impulses ~= 0);

%% Define Zoom Window for Detailed View
symbols_to_display = 20;            % Number of symbols to show in zoom
middle_symbol_index = floor(NUM_DATA_SYMBOLS / 2);
zoom_start_sample = (middle_symbol_index - floor(symbols_to_display / 2)) * ...
                   OVERSAMPLING_RATIO + 1;
zoom_end_sample = zoom_start_sample + symbols_to_display * OVERSAMPLING_RATIO - 1;

% Ensure zoom indices are within signal bounds
zoom_start_sample = max(1, zoom_start_sample);
zoom_end_sample = min(length(transmitted_signal), zoom_end_sample);
zoom_sample_range = zoom_start_sample:zoom_end_sample;

%% Visualization
figure;

% (1) Tx waveform - full
subplot(2,2,1);
plot(1:length(transmitted_signal), transmitted_signal);
title('Transmitted Signal (Full)');
xlabel('Sample Index'); 
ylabel('Amplitude');

% (2) Tx waveform - zoomed
subplot(2,2,2);
plot(zoom_sample_range, transmitted_signal(zoom_sample_range));
title('Transmitted Signal (Zoom: 20 Symbols)');
xlabel('Sample Index'); 
ylabel('Amplitude');

% (3) Matched filter output - full
subplot(2,2,3);
stem(nonzero_indices, symbol_impulses(nonzero_indices), 'filled');
title('Matched Filter Output (Full)');
xlabel('Sample Index'); 
ylabel('Amplitude');

% (4) Matched filter output - zoomed
subplot(2,2,4);
% Find non-zero symbols within zoom range
zoom_nonzero_indices = nonzero_indices(nonzero_indices >= zoom_start_sample & ...
                                       nonzero_indices <= zoom_end_sample);
stem(zoom_nonzero_indices, symbol_impulses(zoom_nonzero_indices), 'filled');
title('Matched Filter Output (Zoom: 20 Symbols)');
xlabel('Sample Index'); 
ylabel('Amplitude');
