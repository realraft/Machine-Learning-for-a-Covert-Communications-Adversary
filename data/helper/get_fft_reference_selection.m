function [freqCenters, fftIndices] = get_fft_reference_selection(numBins, fs, chunkLength, maxAnalysisFreq, referenceMaxBins)
%GET_FFT_REFERENCE_SELECTION Select FFT bins from a provided reference grid.
%   [freqCenters, fftIndices] = GET_FFT_REFERENCE_SELECTION(numBins, fs,
%   chunkLength, maxAnalysisFreq, referenceMaxBins) returns the frequency
%   centers and FFT indices corresponding to evenly spaced samples drawn
%   from a master grid defined by referenceMaxBins.

    binEdges = linspace(0, maxAnalysisFreq, referenceMaxBins + 1);
    binWidth = binEdges(2) - binEdges(1);
    fullCenters = binEdges(1:end-1) + binWidth / 2;

    selection = round(linspace(1, referenceMaxBins, numBins));

    freqCenters = fullCenters(selection);

    df = fs / chunkLength;
    fullIndices = round(fullCenters / df) + 1;
    fullIndices = max(1, min(chunkLength, fullIndices));
    fftIndices = fullIndices(selection);
end
