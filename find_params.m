%% Simulation

%% Parameters

Nsym = 1000; % number of symbols
Nfft = 100; % number of FFT to average
beta = 0.25; % rolloff factor
span = 10; % filter span in symbols
osr = 16; % oversampling rate
Ts = 1; % symbol period
avoid = 10; % number of samples at head and tail to avoid filter transients
a = 1; % initial magnitude guess for symbols (optimization will update)
a1 = 1; % nonlinearity component from Jessica thesis 
a3 = -0.05; % initial nonlinearity guess (optimization will update)
var = 0.1; % noise variance
target_error = 0.4; % desired error rate for the threshold test
rng_seed = 2025; % fixed seed for reproducibility during optimization
num_trials = 3; % number of independent Monte Carlo trials per evaluation

%% Calculations

T = Ts/osr; % sample period
fs = 1/T; % sample rate 
chunk = Nsym * osr; % chunk length

% p(t)
p = rcosdesign(beta, span, osr, 'sqrt'); % square root raised cosine

% pack constant parameters
params = struct('Nsym', Nsym, ...
                'Nfft', Nfft, ...
                'osr', osr, ...
                'avoid', avoid, ...
                'a', a, ...
                'a1', a1, ...
                'a3', a3, ...
                'var', var, ...
                'chunk', chunk, ...
                'rng_seed', rng_seed, ...
                'num_trials', num_trials);

% perform a coarse grid search for a robust starting point
a_grid = linspace(0.5, 6, 12);
a3_grid = linspace(-3, 3, 25);

best_guess = [a; a3];
best_score = inf;

for aa = a_grid
    for cc = a3_grid
        trial_score = thresholdObjective([aa; cc], params, p, target_error);
        if trial_score < best_score
            best_score = trial_score;
            best_guess = [aa; cc];
        end
    end
end

fprintf('Coarse search seed: a = %.3f, a3 = %.3f (score %.4f)\n', best_guess(1), best_guess(2), best_score);

% optimize (a, a3) to reach the target error
objective = @(v) thresholdObjective(v, params, p, target_error);
options = optimset('Display', 'iter', 'TolX', 1e-3, 'TolFun', 1e-3, ...
                   'MaxIter', 100, 'MaxFunEvals', 200);
[v_opt, fval, exitflag, output] = fminsearch(objective, best_guess, options);

params.a = v_opt(1);
params.a3 = v_opt(2);

[error_rate, detection_rate, y_fft_nl, y_fft_l] = evaluateThreshold(params, p);

f = (-chunk/2 : chunk/2-1) * (fs / chunk); % frequency vector from -fs/2 to fs/2
y_fft_nl_sh = fftshift(y_fft_nl);
y_fft_l_sh = fftshift(y_fft_l);

y_fft_nl_db = 10*log10(y_fft_nl_sh + eps);
y_fft_l_db = 10*log10(y_fft_l_sh + eps);

figure;
plot(f, y_fft_l_db, '-b', 'LineWidth', 1.2); hold on;
plot(f, y_fft_nl_db, '--r', 'LineWidth', 1.2);
hold off;
xlabel('Frequency (Hz)');
ylabel('Power (dB)');
title('Average Power Spectral Density of y(t)');
legend('Linear','Nonlinear','Location','best');
grid on;
xlim([-fs/2 fs/2]);

fprintf('Optimization finished with exit flag %d after %d iterations.\n', exitflag, output.iterations);
fprintf('Optimal a = %.4f, a3 = %.4f\n', params.a, params.a3);
fprintf('Detection rate = %.4f, error rate = %.4f (target %.4f)\n', detection_rate, error_rate, target_error);

%% Helper functions

function score = thresholdObjective(v, base_params, p, target_error)
    params = base_params;
    params.a = v(1);
    params.a3 = v(2);

    if params.a <= 0 || ~isfinite(params.a) || ~isfinite(params.a3)
        score = 1e6;
        return;
    end

    [error_rate, ~, ~, ~] = evaluateThreshold(params, p);
    score = (error_rate - target_error)^2;
end

function [error_rate, detection_rate, y_fft_nl_avg, y_fft_l_avg] = evaluateThreshold(params, p)
    Nsym = params.Nsym;
    Nfft = params.Nfft;
    osr = params.osr;
    avoid = params.avoid;
    a = params.a;
    a1 = params.a1;
    a3 = params.a3;
    var = params.var;
    chunk = params.chunk;
    num_trials = params.num_trials;

    y_fft_nl_total = zeros(1, chunk);
    y_fft_l_total = zeros(1, chunk);

    for trial = 1:num_trials
        rng(params.rng_seed + trial - 1);

        y_fft_nl = zeros(1, chunk);
        y_fft_l = zeros(1, chunk);

        for k = 1:Nfft
            ak = 2*a*randi([0 1], 1, Nsym) - a;
            ak = upsample(ak, osr);
            ak(1:avoid) = 0;
            ak(end-avoid+1:end) = 0;

            x_linear = conv(ak, p, 'same');
            x_nl = a1*x_linear + a3*x_linear.^3;

            noise = sqrt(var) * randn(size(x_linear));

            x_linear_noisy = x_linear + noise;
            x_nl_noisy = x_nl + noise;

            y_linear = conv(x_linear_noisy, p, 'same');
            y_nl = conv(x_nl_noisy, p, 'same');

            fft_linear = fft(y_linear, chunk);
            fft_nl = fft(y_nl, chunk);

            y_fft_l = y_fft_l + abs(fft_linear).^2;
            y_fft_nl = y_fft_nl + abs(fft_nl).^2;
        end

        y_fft_nl = y_fft_nl / Nfft;
        y_fft_l = y_fft_l / Nfft;

        y_fft_nl_total = y_fft_nl_total + y_fft_nl;
        y_fft_l_total = y_fft_l_total + y_fft_l;
    end

    y_fft_nl_avg = y_fft_nl_total / num_trials;
    y_fft_l_avg = y_fft_l_total / num_trials;

    detection_mask = y_fft_nl_avg > y_fft_l_avg;
    detection_rate = mean(detection_mask);
    error_rate = 1 - detection_rate;
end
