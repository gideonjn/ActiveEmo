function [ norm_features ] = zNormalize( features, speakers )

    uniqueSpeakers = unique(speakers);
    meanVals = features;
    stdVals = features;
    for i = 1:numel(uniqueSpeakers)
        sOn = uniqueSpeakers(i);
        ind = find(speakers==sOn);
        meanVals(ind,:)= repmat(mean(features(ind,:)),size(ind,1),1);
        % Check for flat line
        stdDiv = repmat(std(features(ind,:)),size(ind,1),1);
        stdDiv(stdDiv==0) = 1;
        stdVals(ind,:) = stdDiv;    
    end
    norm_features = (features - meanVals) ./ stdVals;

end
