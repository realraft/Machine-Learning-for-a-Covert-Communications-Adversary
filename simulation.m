clc; clear; close all;

%% Parameters
Ts = 1;              % Symbol period
OSR = 16;            % Oversampling rate
T = Ts / OSR;        % Discrete-time step
N_symbols = 1000;    % Number of symbols
N_zeros = 10;        % Number of zeros at start and end
rolloff = 0.25;      % SRRC roll-off factor
span = 6;            % Filter span in symbols

%% Symbols a_k
a_k = randi([0 1], 1, N_symbols)*2 - 1;   % random ±1
a_k(1:N_zeros) = 0;                       % first zeros
a_k(end-N_zeros+1:end) = 0;               % last zeros

%% Square-root raised cosine pulse
p = rcosdesign(rolloff, span, OSR, 'sqrt');  

%% x(nT) using summation
x = zeros(1, (N_symbols-1)*OSR + length(p));
for k = 1:N_symbols
    t_index = (0:length(p)-1) + (k-1)*OSR;
    x(t_index+1) = x(t_index+1) + a_k(k)*p;
end

%% Time vector matching x
nT = (0:length(x)-1)*T;

%% Full figure with two subplots
figure('Position', [100, 100, 1200, 600]);

% --- Subplot 1: full signal ---
subplot(2,1,1);
plot(nT, x, 'LineWidth', 1.2);
xlabel('Time');
ylabel('x(nT)');
title('Full discrete-time signal with SRRC pulse shaping');
grid on;

% --- Subplot 2: zoomed-in section in the middle ---
symbols_to_show = 20;                               % number of symbols to zoom
mid_symbol = floor(N_symbols/2);                    % middle symbol index
start_index = (mid_symbol - floor(symbols_to_show/2))*OSR + 1;
end_index = start_index + symbols_to_show*OSR + length(p) - 1;

subplot(2,1,2);
plot(nT(start_index:end_index), x(start_index:end_index), 'LineWidth', 1.5);
xlabel('Time');
ylabel('x(nT)');
title(['Zoom: ', num2str(symbols_to_show), ' symbols in the middle']);
grid on;