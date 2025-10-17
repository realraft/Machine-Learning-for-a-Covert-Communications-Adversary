function [avgSpectrum, yTime] = compute_fft_average(params, pulseShape, useNonlinearity, useNoise)
%COMPUTE_FFT_AVERAGE Average |FFT|^2 of the matched-filter output.
%   params struct expects fields:
%     Nsym, Nfft, osr, avoid, a, a1, a3, noiseVar
%   pulseShape is the pulse shaping filter.
%   useNonlinearity/useNoise toggle the extra effects.

    Nsym = params.Nsym;
    Nfft = params.Nfft;
    osr = params.osr;
    avoid = params.avoid;
    a = params.a;
    a1 = params.a1;
    a3 = params.a3;
    noiseVar = params.noiseVar;

    chunk = Nsym * osr;
    pulseRow = pulseShape(:).';

    avgSpectrum = zeros(1, chunk);
    yTime = zeros(1, chunk);

    for k = 1:Nfft
        ak = 2 * a * randi([0, 1], 1, Nsym) - a;
        ak = upsample(ak, osr);
        ak(1:avoid) = 0;
        ak(end-avoid+1:end) = 0;

        x = conv(ak, pulseRow, 'same');

        if useNonlinearity
            x = a1 * x + a3 * (x .^ 3);
        end

        if useNoise
            x = x + sqrt(noiseVar) * randn(size(x));
        end

        yTime = conv(x, pulseRow, 'same');

        fftChunk = fft(yTime, chunk);
        avgSpectrum = avgSpectrum + abs(fftChunk) .^ 2;
    end

    avgSpectrum = avgSpectrum / Nfft;
end
