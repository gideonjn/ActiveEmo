function [ gain ] = calcContGain( vals, labels )
    
    nVals = numel(vals);
    maxBuckets = min([8 nVals]);
    [sVals ind] = sort(vals);
    sortedLabels = labels(ind);
    gain = 0;
    for bSize = 2:maxBuckets
        divVals = zeros(size(sVals));
        for i = 2:nVals
            if sVals(i) == sVals(i-1)
                divVals(i) = divVals(i-1);
            else
                divVals(i) = floor(((i-1)/nVals)*bSize);
            end
        end
        
        curGain = calcGain(divVals, sortedLabels);
        if curGain > gain
            gain = curGain;
        end
    end
    
end

