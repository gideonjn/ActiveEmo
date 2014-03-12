%% Load Data
dataPath = 'C:\Research\';
%dataPath = '/home/gideonjn/';
dsName = 'Haberman';
[allFeaturesInit, allLabelsInit, k, nKIt, nTest, nIt, sigma] = ...
    importData(dataPath, dsName);
nSamples = numel(allLabelsInit);

%% Determine uniques
allIdLabels = allLabelsInit;
labelToId = unique(allLabelsInit);
nClasses = numel(labelToId);
for i = 1:nClasses
    allIdLabels(allIdLabels==i) = i-1;
end 

%% Initialize Runs
rng(0); % randomness control
maxTime = 50;
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

    % Get clusters
    Tr0_features = Tr_features(Tr_labels==0,:);
    Tr1_features = Tr_features(Tr_labels==1,:);
    ind0 = kmeans(Tr0_features, k, 'replicates', nKIt);
    ind1 = kmeans(Tr1_features, k, 'replicates', nKIt);
    L_features = [];
    L_labels = [];

    % Get middle point from each cluster
    for i = 1:k
        mid0 = mean(Tr0_features(ind0==i,:),1);
        [maxVal maxInd] = max(sum((Tr0_features-repmat(mid0,size(Tr0_features,1),1)).^2,2));
        indEx0 = maxInd(1);
        L_features = vertcat(L_features, Tr0_features(indEx0,:));
        L_labels = vertcat(L_labels, 0);
        Tr0_features(indEx0,:) = [];
        ind0(indEx0) = [];
        
        mid1 = mean(Tr1_features(ind1==i,:),1);
        [maxVal maxInd] = max(sum((Tr1_features-repmat(mid1,size(Tr1_features,1),1)).^2,2));
        indEx1 = maxInd(1);
        L_features = vertcat(L_features, Tr1_features(indEx1,:));
        L_labels = vertcat(L_labels, 1);
        Tr1_features(indEx1,:) = [];
        ind1(indEx1) = [];
    end
    
    % Divide into labelled/unlabelled set
    U_features = vertcat(Tr0_features, Tr1_features);
    U_labels = vertcat(zeros(size(Tr0_features,1),1), ...
        ones(size(Tr1_features,1),1));
    rp = randperm(size(U_labels,1)); % random permutation of indices
    U_features = U_features(rp,:);
    U_labels = U_labels(rp);
    rp = randperm(size(L_labels,1)); % random permutation of indices
    L_features = L_features(rp,:);
    L_labels = L_labels(rp);

    % Run tests
    train_args = ['-t 2 -g ' num2str(sigma) ' -c 8192 -q -b 1'];
    test_args = '-b 1 -q';
    nStar = 1;
    numUn = size(U_labels,1);
    for step = 1:maxTime
        tic;
        % Assume all unlabeled are 0
        assume0_features = vertcat(L_features, U_features);
        assume0_labels = vertcat(L_labels, zeros(size(U_features,1),1));
        model = libsvmtrain(assume0_labels, assume0_features, train_args);
        [~, ~, prob0] = libsvmpredict(U_labels, U_features, model, test_args);
        dist0 = log(prob0(:,1)./(1-prob0(:,1)));

        % Assume all unlabeled are 1
        assume1_features = vertcat(L_features, U_features);
        assume1_labels = vertcat(L_labels, ones(size(U_features,1),1));
        model = libsvmtrain(assume1_labels, assume1_features, train_args);
        [~, ~, prob1] = libsvmpredict(U_labels, U_features, model, test_args);
        dist1 = log(prob1(:,1)./(1-prob1(:,1)));

        % Calculate Inconsistency
        sumDist = abs(dist0) + abs(dist1);
        pr0 = abs(dist0) ./ sumDist;
        pr1 = abs(dist1) ./ sumDist;
        incon = -(pr0.*log2(pr0))-(pr1.*log2(pr1));

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
        cMtrx = confusionmat(T_labels, y_hat);
        UnAcc = ((cMtrx(1,1)/sum(cMtrx(1,:)))+(cMtrx(2,2)/sum(cMtrx(2,:))))./2;
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