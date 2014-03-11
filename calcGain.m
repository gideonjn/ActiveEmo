function [ gain ] = calcGain( vals, labels )

    nVals = numel(vals);
    uVals = unique(vals);
    remainder = 0;
    for uOn = 1:numel(uVals)
        probVal = sum(uVals(uOn)==vals) ./ nVals;
        remainder = remainder + (probVal.*calcEntropy(labels(uVals(uOn)==vals)));
    end
    gain = calcEntropy(labels) - remainder;

end

