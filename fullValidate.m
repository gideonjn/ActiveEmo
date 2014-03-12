function [ maxG, maxC, featureRange, maxUnAcc ] = fullValidate( Tr_features, Tr_labels, fSubsetSize)

    % Determine best features from info gain over all training data
    nTrain = numel(Tr_labels);
    unLabels = unique(Tr_labels);
    nClasses = numel(unLabels);
    nF = size(Tr_features,2);
    gain = zeros(1,nF);
    for fOn = 1:nF
        gain(fOn) = calcContGain(Tr_features(:,fOn), Tr_labels);
    end
    [sG ind] = sort(gain, 'descend');
    featureRank = ind;
    featureRange = featureRank(1:fSubsetSize);
    Tr_features = Tr_features(:,featureRange);
    
    % Get validation set taken from all speakers
    nVal = ceil(nTrain/5);
    rp = randperm(nTrain); % random permutation of indices
    Tr_features = Tr_features(rp,:);
    Tr_labels = Tr_labels(rp);
    Val_features = Tr_features(1:nVal,:);
    Val_labels = Tr_labels(1:nVal);
    Tr2_features = Tr_features(nVal+1:end,:);
    Tr2_labels = Tr_labels(nVal+1:end,:);
    
    % Determine ideal parameters using validation set
    maxUnAcc = -1;
    for cExp = -5:2:15 
        for gExp = -15:2:3
            cVal = 2.^cExp;
            gVal = 2.^gExp;
            train_args = ['-t 2 -q -g ' num2str(gVal) ' -c ' num2str(cVal)];
            test_args = ['-q'];
            % Determine validation accuracy
            model = libsvmtrain(Tr2_labels, Tr2_features, train_args);
            [y_hat, ~, ~] = libsvmpredict(Val_labels, Val_features, model, test_args);
            cMtrx = confusionmat(Val_labels, y_hat, 'order', 1:nClasses);
            cPct = cMtrx./repmat(max(sum(cMtrx,2),1),1,nClasses);
            UnAcc = mean(sum(eye(nClasses,nClasses).*cPct));              
            if UnAcc > maxUnAcc
                maxUnAcc = UnAcc;
                maxC = cVal;
                maxG = gVal;
            end
        end
    end
   
end

