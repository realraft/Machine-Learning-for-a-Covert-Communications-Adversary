function stats = compute_spectral_stats(analysisFreqs, analysisPower)
%% Compute summary statistics for a power spectrum slice.
%   stats = COMPUTE_SPECTRAL_STATS(analysisFreqs, analysisPower) returns
%   aggregate metrics used throughout the data generation scripts.

    stats = struct();
    stats.powerSum = sum(analysisPower) + eps;
    stats.maxPower = max(analysisPower);
    stats.psd = analysisPower / stats.powerSum;

    stats.entropy = -sum(stats.psd .* log2(stats.psd + eps));

    stats.centroid = sum(analysisFreqs .* analysisPower) / stats.powerSum;
    stats.centroidNorm = stats.centroid / (max(analysisFreqs) + eps);

    freqDiff = analysisFreqs - stats.centroid;
    spectralStd = sqrt(sum((freqDiff .^ 2) .* stats.psd) + eps);

    stats.skewness = sum((freqDiff .^ 3) .* stats.psd) / (spectralStd ^ 3 + eps);
    stats.kurtosis = sum((freqDiff .^ 4) .* stats.psd) / (spectralStd ^ 4 + eps);

    stats.flatness = exp(mean(log(analysisPower + eps))) / (mean(analysisPower) + eps);
end
