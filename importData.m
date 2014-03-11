function [ features, labels, k, nKIt, nTest, nIt, sigma ] = importData( path, name )

switch name
    case 'Australian'
        k = 5;
        nKIt = 100;
        nTest = 138;
        nIt = 50;
        sigma = 1;
    case 'Haberman'
        k = 2;
        nKIt = 100;
        nTest = 61;
        nIt = 50;
        sigma = 1;
        data = csvread([path 'Data/ActiveEmo/Haberman/haberman.data']);
        features = data(:,1:3);
        labels = data(:,4);
    case 'Soybean'
        k = 5;
        nKIt = 100;
        nTest = 137;
        nIt = 50;
        sigma = 1;
    case 'Pima'
        k = 10;
        nKIt = 500;
        nTest = 154;
        nIt = 50;
        sigma = 1;
        data = csvread([path 'Data/ActiveEmo/Pima/pima-indians-diabetes.data']);
        features = data(:,1:8);
        labels = data(:,9);
    case 'Ecoli'
        k = 5;
        nKIt = 100;
        nTest = 67;
        nIt = 50;
        sigma = 1;
    case 'Yeast'
        k = 3;
        nKIt = 100;
        nTest = 297;
        nIt = 50;
        sigma = 1;
    case 'Spam'
        k = 10;
        nKIt = 1000;
        nTest = 920;
        nIt = 10;
        sigma = 32;
        data = csvread([path 'Data/ActiveEmo/Spam/spambase.data']);
        features = data(:,1:57);
        labels = data(:,58);
    case 'Optdigits'
        k = 10;
        nKIt = 100;
        nTest = 1797;
        nIt = 10;
        sigma = 32;
    case 'EmoDB'
        k = 5;
        nKIt = 100;
        nIt = 10;
        sigma = 0.125;
        data = importdata([path 'emote/Features/EmoDB_labels.csv']);
        labels = data(2,:)';
        speakers = data(1,:)';
        data = importdata([path 'emote/Features/EmoDB_features.csv']);
        features = data.data(:,3:end);

        % EXC to HAP
        labels(labels==6)=5;

        % Remove all but HAP, SAD, ANG, NEU
        ind = find(labels==5 | labels==3);
        labels = labels(ind,:);
        speakers = speakers(ind,:);
        features = features(ind,:);

        nTest = numel(labels)/5;

        uniqueSpeakers = unique(labels);
        meanVals = features;
        stdVals = features;
        for i = 1:numel(uniqueSpeakers)
            sOn = uniqueSpeakers(i);
            ind = find(labels==sOn);
            meanVals(ind,:)= repmat(mean(features(ind,:)),size(ind,1),1);
            stdVals(ind,:) = repmat(std(features(ind,:)),size(ind,1),1);
        end
        features = (features - meanVals) ./ stdVals;

        nF = size(features,2);
        gain = zeros(1,nF);
        for fOn = 1:nF
            gain(fOn) = calcContGain(features(:,fOn), labels);
        end
        [sG ind] = sort(gain, 'descend');
        featureRank = ind;
        features = features(:,featureRank(1:20));
        
        nTest = ceil(numel(labels)/5);
end

% Convert label strings into integers
% allLabels = zeros(nSamples,1);
% labelNames = unique(labelStr);
% nLabelNames = size(labelNames,1);
% for i=1:nSamples
%     for j=1:nLabelNames
%         if strcmp(labelNames{j}, labelStr{i})
%             allLabels(i) = j;
%             continue;
%         end
%     end
% end

end

