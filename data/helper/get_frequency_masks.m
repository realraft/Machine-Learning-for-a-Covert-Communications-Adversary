function masks = get_frequency_masks(analysisFreqs)
%% Return logical masks for standard frequency regions.
%   masks = GET_FREQUENCY_MASKS(analysisFreqs) provides reusable frequency
%   band definitions for the generated spectra.

    masks = struct();
    masks.main = analysisFreqs <= 1;
    masks.shoulder = analysisFreqs > 1 & analysisFreqs <= 2.5;
    masks.outer = analysisFreqs > 2.5;
    masks.adjacent = analysisFreqs > 1 & analysisFreqs <= 2;
    masks.intermod = analysisFreqs > 1.5 & analysisFreqs <= 3;
    masks.harmonic = analysisFreqs >= 2 & analysisFreqs <= max(analysisFreqs);
    masks.regrowth = analysisFreqs >= 3 & analysisFreqs <= max(analysisFreqs);
end
