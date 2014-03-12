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
        k = 3;
        nKIt = 100;
        nTest = 67;
        nIt = 50;
        sigma = 1;
        name = [path 'Data/ActiveEmo/Ecoli/ecoli.data'];
        [features labels] = importTextData(name, 2:8, 9);
        nClasses = max(labels);
        nums = hist(labels,1:nClasses);
        for i = 1:nClasses
            if nums(i) < k*3
                features(labels==i,:) = [];
                labels(labels==i,:) = [];
            end
        end
    case 'Yeast'
        k = 3;
        nKIt = 100;
        nTest = 297;
        nIt = 50;
        sigma = 1;
        name = [path 'Data/ActiveEmo/Yeast/yeast.data'];
        [features labels] = importTextData(name, 2:9, 10);
        nClasses = max(labels);
        nums = hist(labels,1:nClasses);
        for i = 1:nClasses
            if nums(i) < k*3
                features(labels==i,:) = [];
                labels(labels==i,:) = [];
            end
        end
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
        dataTes = csvread([path 'Data/ActiveEmo/Optdigits/optdigits.tes']);
        dataTra = csvread([path 'Data/ActiveEmo/Optdigits/optdigits.tra']);
        features = vertcat(dataTes(:,1:64), dataTra(:,1:64));
        labels = vertcat(dataTes(:,65), dataTra(:,65));
end

end

