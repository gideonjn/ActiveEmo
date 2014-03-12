function [ selected ] = selectAL( L_features, L_labels, U_features, params )

    % Set up parameters
    train_args = ['-t 2 -g ' num2str(params.maxG) ' -c ' num2str(params.maxC) ' -q -b 1'];
    test_args = '-b 1 -q';
    nClasses = numel(unique(L_labels));
    
    % Assume all unlabeled are each class
    dist = zeros(size(U_features,1),nClasses);
    assumeFeatures = vertcat(L_features, U_features);
    for classOn = 1:nClasses
        assumeLabels = vertcat(L_labels, ones(size(U_features,1),1).*classOn);
        model = libsvmtrain(assumeLabels, assumeFeatures, train_args);
        U_labels = zeros(size(U_features,1),1);
        [~, ~, probSVM] = libsvmpredict(U_labels, U_features, model, test_args);
        dist(:,classOn) = log(probSVM(:,1)./(1-probSVM(:,1)));
    end

    % Calculate Inconsistency
    sumDist = sum(abs(dist),2);
    prob = abs(dist) ./ repmat(sumDist,1,nClasses);
    incon = -sum(prob.*log2(prob),2);

    % Select nStar highest inconsistency points
    [order maxInd] = sort(incon, 'descend');
    selected = maxInd(1:params.numToSelect);

end
