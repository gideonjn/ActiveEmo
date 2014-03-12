function [ features, labels ] = importTextData( file, featureCol, labelCol )
    data = importdata(file);
    nSamples = numel(data);
    nFeatures = numel(featureCol);
    features = zeros(nSamples, nFeatures);
    labels = zeros(nSamples,1);
    foundLabels = cell(0,0);
    for i = 1:nSamples
        splStr = regexp(data(i),'\s+','split');
        for f = 1:nFeatures
            features(i,f) = str2double(splStr{1}{featureCol(f)});
        end
        label = splStr{1}{labelCol};
        found = 0;
        for j = 1:numel(foundLabels)
            if strcmp(foundLabels{j}, label)
                labels(i) = j;
                found = 1;
                break;
            end
        end
        if found == 0
            labels(i) = numel(foundLabels)+1;
            foundLabels{numel(foundLabels)+1} = label;
        end
    end
        
end