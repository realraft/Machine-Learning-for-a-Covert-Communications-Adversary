function names = format_fft_feature_names(freqCenters)
%FORMAT_FFT_FEATURE_NAMES Generate frequency-based FFT feature labels (Hz).
%   names = FORMAT_FFT_FEATURE_NAMES(freqCenters) returns formatted labels
%   such as 'fft_0.125Hz' that correspond to the provided frequency centers
%   (in Hz). The shape of the output matches the input.

    outputSize = size(freqCenters);
    freqCenters = freqCenters(:);

    scaledCenters = round(freqCenters, 6);
    formatted = string(compose('%.3f', scaledCenters));
    formatted = regexprep(formatted, '0+$', '');
    formatted = regexprep(formatted, '\.$', '');
    formatted(formatted == "") = "0";

    names = cellstr("fft_" + formatted + "Hz");
    names = reshape(names, outputSize);
end
