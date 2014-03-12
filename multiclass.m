%% Load Data
%dataPath = 'C:\Research\';
dataPath = '/home/gideonjn/';
dsName = 'Spam';
[allFeaturesInit, allLabelsInit, k, kIt, nTest, nIt, sigma] = ...
    importData(dataPath, dsName);
nSamples = numel(allLabelsInit);
fprintf('Starting run with dataset %s\n', dsName)

%% Determine uniques
allIdLabels = allLabelsInit;
labelToId = unique(allLabelsInit);
nClasses = numel(labelToId);
for i = 1:nClasses
    allIdLabels(allLabelsInit==labelToId(i)) = i;
end 

%% Initialize Runs
rng(0); % randomness control
maxTime = 100;
allAcc = zeros(nIt,maxTime);
allUnAcc = zeros(nIt,maxTime);
allDur = zeros(nIt,maxTime);

%% Run Tests
for itOn = 1:nIt
    rp = randperm(nSamples); % random permutation of indices
    allFeatures = allFeaturesInit(rp,:);
    allLabels = allIdLabels(rp);
    T_features = allFeatures(1:nTest,:);
    T_labels = allLabels(1:nTest);
    Tr_features = allFeatures(nTest+1:end,:);
    Tr_labels = allLabels(nTest+1:end,:);
    nTrain = size(Tr_labels,1);

    % Z-normalize all features
    means = mean(Tr_features);
    stds = std(Tr_features);
    stds(stds==0) = 1;
    for i = 1:nTrain
        Tr_features(i,:) = (Tr_features(i,:) - means)./stds;
    end
    for i = 1:nTest
        T_features(i,:) = (T_features(i,:) - means)./stds;
    end

    % Get cluster ranges
    fullRange = randperm(nTrain);
    labeledRange = getLabeled(Tr_features, Tr_labels, k, kIt);
    fullRange(labeledRange) = [];
    unlabeledRange = fullRange;
    
    % Divide into labelled/unlabelled set
    L_features = Tr_features(labeledRange,:);
    L_labels = Tr_labels(labeledRange);
    U_features = Tr_features(unlabeledRange,:);
    U_labels = Tr_labels(unlabeledRange);

    % Run tests
    train_args = ['-t 2 -g ' num2str(sigma) ' -c 100 -q -b 1'];
    test_args = '-b 1 -q';
    nStar = 1;
    numUn = size(U_labels,1);
    for step = 1:maxTime
        tic;
        % Assume all unlabeled are each class
        dist = zeros(numel(U_labels),nClasses);
        assumeFeatures = vertcat(L_features, U_features);
        for classOn = 1:nClasses
            assumeLabels = vertcat(L_labels, ones(size(U_features,1),1).*classOn);
            model = libsvmtrain(assumeLabels, assumeFeatures, train_args);
            [~, ~, probSVM] = libsvmpredict(U_labels, U_features, model, test_args);
            dist(:,classOn) = log(probSVM(:,1)./(1-probSVM(:,1)));
        end

        % Calculate Inconsistency
        sumDist = sum(abs(dist),2);
        prob = abs(dist) ./ repmat(sumDist,1,nClasses);
        incon = -sum(prob.*log2(prob),2);

        % Select nStar highest inconsistency points
        [order maxInd] = sort(incon, 'descend');
        moveInd = maxInd(1:nStar);
        L_features = vertcat(L_features, U_features(moveInd,:));
        L_labels = vertcat(L_labels, U_labels(moveInd,:));
        U_features(moveInd,:) = [];
        U_labels(moveInd) = [];

        % Determine test accuracy
        model = libsvmtrain(L_labels, L_features, train_args);
        [y_hat, Acc, ~] = libsvmpredict(T_labels, T_features, model, test_args);
        allAcc(itOn,step) = Acc(1);
        cMtrx = confusionmat(T_labels, y_hat, 'order', 1:nClasses);
        cPct = cMtrx./repmat(sum(cMtrx,2),1,nClasses);
        UnAcc = mean(sum(eye(nClasses,nClasses).*cPct));        
        allUnAcc(itOn,step) = UnAcc;
        fprintf('Iteration %d Step %d Unweighted Accuracy = %.4f%% Weighted Accuracy = %.4f%%\n', ...
            itOn, step, UnAcc*100, Acc(1));
        allDur(itOn,step) = toc;
    end
end

%% Save state
save([dsName '.mat']);

%% Load state
load([dsName '.mat']);

%% Average plot - Weighted Accuracy
close all;
h = figure;
plotAcc = mean(allAcc,1);
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
for itOn = 1:nIt
    plot(allAcc(itOn,:));
end
hold off
ylabel('Weighted Accuracy (Percent)');
xlabel('Iteration');
saveas(h,['Images/' dsName '_AllAcc'],'png');