function [ Acc UnAcc ] = fullTest( Tr_features, Tr_labels, ...
    T_features, T_labels, maxC, maxG )

    nClasses = numel(unique(vertcat(Tr_labels, T_labels)));
    train_args = ['-t 2 -q -g ' num2str(maxG) ' -c ' num2str(maxC)];
    test_args = ['-q'];
    model = libsvmtrain(Tr_labels, Tr_features, train_args);
    [y_hat, accMtrx, ~] = libsvmpredict(T_labels, T_features, model, test_args);
    Acc = accMtrx(1)/100;
    cMtrx = confusionmat(T_labels, y_hat, 'order', 1:nClasses);
    cPct = cMtrx./repmat(sum(cMtrx,2),1,nClasses);
    UnAcc = mean(sum(eye(nClasses,nClasses).*cPct));

end

