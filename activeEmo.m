%% Load Data
if isunix
    dataPath = '/home/gideonjn/';
else
    dataPath = 'C:\Research\';
end
dsName = 'EmoDB';
[allFeaturesInit, allLabelsInit, allSpeakersInit] = importEmo(dataPath, dsName);

%% Only allow certain labels
% EXC to HAP
allLabelsInit(allLabelsInit==6)=5;
% Remove all but HAP, SAD, ANG, NEU
ind = find(allLabelsInit==0 | allLabelsInit==3 | ...
        allLabelsInit==4 | allLabelsInit==5);
allLabelsInit = allLabelsInit(ind,:);
allSpeakersInit = allSpeakersInit(ind,:);
allFeaturesInit = allFeaturesInit(ind,:);
nSamples = numel(allLabelsInit);

%% Determine label uniques
allIdLabels = allLabelsInit;
labelToId = unique(allLabelsInit);
nClasses = numel(labelToId);
for i = 1:nClasses
    allIdLabels(allLabelsInit==labelToId(i)) = i;
end 

%% Determine speaker uniques
allIdSpeakers = allSpeakersInit;
speakerToId = unique(allSpeakersInit);
nSpeakers = numel(speakerToId);
for i = 1:nSpeakers
    allIdSpeakers(allSpeakersInit==speakerToId(i)) = i;
end 

%% Z-normalize features by speaker
allFeaturesInit = zNormalize(allFeaturesInit, allSpeakersInit);

%% Set test parameters
params.maxTime = 100;
params.fSubsetSize = 20;
params.runAllDataTest = 0;
params.validateAL = 1;
params.numToSelect = 1;
params.k = 5;
params.kIt = 100;

%% Initialize Runs
rng(0); % randomness control
allAcc = zeros(nSpeakers,params.maxTime);
allUnAcc = zeros(nSpeakers,params.maxTime);
allDur = zeros(nSpeakers,params.maxTime);

%% Run Tests over each speaker
for sOn = 1:nSpeakers
    % Separate training and testing data
    rp = randperm(nSamples); % random permutation of indices
    allFeatures = allFeaturesInit(rp,:);
    allLabels = allIdLabels(rp);
    allSpeakers = allIdSpeakers(rp);
    T_features = allFeatures(allSpeakers==sOn,:);
    T_labels = allLabels(allSpeakers==sOn,:);
    Tr_features = allFeatures(allSpeakers~=sOn,:);
    Tr_labels = allLabels(allSpeakers~=sOn,:);
    nTrain = numel(Tr_labels);
    
    % Run validation and test with data entirely known
    if (params.runAllDataTest)
        [ params.maxG, params.maxC, featureRange, ~ ] = ...
            fullValidate( Tr_features, Tr_labels, fSubsetSize);
        [Acc UnAcc] = fullTest(Tr_features(:,featureRange), Tr_labels, ...
            T_features(:,featureRange), T_labels, maxC, maxG);
    end
    
    % Get cluster ranges
    fullRange = randperm(nTrain);
    labeledRange = getLabeledSet(Tr_features, Tr_labels, params.k, params.kIt);
    fullRange(labeledRange) = [];
    unlabeledRange = fullRange;
    
    % Divide into labelled/unlabelled set
    L_features = Tr_features(labeledRange,:);
    L_labels = Tr_labels(labeledRange);
    U_features = Tr_features(unlabeledRange,:);
    U_labels = Tr_labels(unlabeledRange);

    % Use active learning to iteratively add new samples
    for step = 1:params.maxTime
        tic;
        % Get SVM parameters
        if (params.validateAL && step==1)
            [ params.maxG, params.maxC, featureRange, ValUnAcc ] = ...
                fullValidate( L_features, L_labels, params.fSubsetSize);
        end

        % Select points using AL
        selected = selectAL(L_features(:,featureRange), L_labels, U_features(:,featureRange), params);
        L_features = vertcat(L_features, U_features(selected,:));
        L_labels = vertcat(L_labels, U_labels(selected,:));
        U_features(selected,:) = [];
        U_labels(selected) = [];
        
        % Get SVM parameters
        if (params.validateAL)
            [ params.maxG, params.maxC, featureRange, ValUnAcc ] = ...
                fullValidate( L_features, L_labels, params.fSubsetSize);
        end
        
        % Determine test accuracy
        [Acc UnAcc] = fullTest(Tr_features(:,featureRange), Tr_labels, ...
            T_features(:,featureRange), T_labels, params.maxC, params.maxG);
        allAcc(sOn,step) = Acc;        
        allUnAcc(sOn,step) = UnAcc;
        fprintf('Speaker %d Step %d Unweighted Accuracy = %.4f%% Weighted Accuracy = %.4f%%\n', ...
            sOn, step, UnAcc*100, Acc*100);
        allDur(sOn,step) = toc;
    end
end

%% Save state
save([dsName '.mat']);

%% Load state
load([dsName '.mat']);

%% Average plot - Weighted Accuracy
close all;
h = figure;
plotAcc = mean(allAcc.*100,1);
plot(plotAcc);
ylabel('Weighted Accuracy (Percent)');
xlabel('Iteration');
saveas(h,['Images/' dsName '_Acc'],'png');

%% Average plot - Unweighted Accuracy
close all;
h = figure;
plotUnAcc = mean(allUnAcc.*100,1);
plot(plotUnAcc);
ylabel('Unweighted Accuracy (Percent)');
xlabel('Iteration');
saveas(h,['Images/' dsName '_UnAcc'],'png');

%% Average plot - Deviation
close all;
h = figure;
plotDev = abs(plotAcc - mean(plotAcc,2));
plot(plotDev);
ylabel('Deviation from Mean Accuracy');
xlabel('Iteration');
saveas(h,['Images/' dsName '_Dev'],'png');

%% Average plot - Duration
close all;
h = figure;
plotDur = mean(allDur,1);
plot(plotDur);
ylabel('Duration of Iteration (Seconds)');
xlabel('Iteration');
saveas(h,['Images/' dsName '_Dur'],'png');

%% All plots - Weighted Accuracy
close all;
h = figure;
clf
hold on
for itOn = 1:nSpeakers
    plot(allAcc(itOn,:).*100);
end
hold off
ylabel('Weighted Accuracy (Percent)');
xlabel('Iteration');
saveas(h,['Images/' dsName '_AllAcc'],'png');
