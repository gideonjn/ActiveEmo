function [ features, labels, speakers ] = importEmo( path, name )

    data = importdata([path 'emote/Features/' name '_labels.csv']);
    labels = data(2,:)';
    speakers = data(1,:)';
    data = importdata([path 'emote/Features/' name '_features.csv']);
    features = data.data(:,3:end);
    
end

