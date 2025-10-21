function write_feature_table(featureMatrix, labels, targetFreqs, extraNames, outputFile)
%% Convert features to a table and write them to disk.
%   WRITE_FEATURE_TABLE(featureMatrix, labels, targetFreqs, extraNames,
%   outputFile) adds formatted FFT feature names, appends labels, and
%   writes the table to the requested location.

    if nargin < 4 || isempty(extraNames)
        extraNames = {};
    end

    fftNames = format_fft_feature_names(targetFreqs);
    featureNames = [fftNames, extraNames];

    featureTable = array2table(featureMatrix, 'VariableNames', featureNames);
    featureTable.label = labels;

    writetable(featureTable, outputFile);
    fprintf('Saved %d rows to %s\n', size(featureMatrix, 1), outputFile);
end
