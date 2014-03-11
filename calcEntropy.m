function [ entropy ] = calcEntropy( vals )

    nVals = numel(vals);
    uVals = unique(vals);
    entropy = 0;
    if numel(uVals) ~= 1 
        for uOn = 1:numel(uVals)
            probVal = sum(uVals(uOn)==vals) ./ nVals;
            entropy = entropy + probVal.*log(probVal)/log(2);
        end
        entropy = -entropy;
    end

end

