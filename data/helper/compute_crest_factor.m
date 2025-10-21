function value = compute_crest_factor(signal)
%% Compute crest factor for a time-domain waveform.
%   value = COMPUTE_CREST_FACTOR(signal) returns the peak-to-RMS ratio for
%   the provided real sequence.

    rmsVal = sqrt(mean(signal .^ 2) + eps);
    value = max(abs(signal)) / rmsVal;
end
