function [ labelRange ] = getLabeledSet( features, labels, k, kIt )
    
    % Get important stats - Assumes labels go from 1-n
    nClasses = max(labels);
    labelRange = [];
    
    % Get k clusters from each class
    for classOn = 1:nClasses
        % Perform k-means
        subInd = find(labels==classOn);
        subFeatures = features(subInd,:);
        clusterId = kmeans(subFeatures, k, 'replicates', kIt);
        
        % Get midpoint of each cluster
        for i = 1:k
            indCluster = find(clusterId==i);
            clusterFeatures = subFeatures(indCluster,:);
            mid = mean(clusterFeatures,1);
            midDiff = repmat(mid,size(clusterFeatures,1),1);
            clusterDist = sum((clusterFeatures-midDiff).^2,2);
            [maxVal maxInd] = max(clusterDist);
            indCenter = subInd(indCluster(maxInd(1)));
            labelRange = [labelRange indCenter];
        end
    end
    
    % Shuffle the range
    labelRange = labelRange(randperm(numel(labelRange)));

end
