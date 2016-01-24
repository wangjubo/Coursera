load('ex6data3.mat');

possibleValue = [0.01 0.03 0.1 0.3 1 3 10 30];
[m n] = size(possibleValue);
error = zeros(n);
for i = 1 : n
    for j = 1 : n
        % Train the SVM
        model= svmTrain(X, y, possibleValue(1,i), @(x1, x2) gaussianKernel(x1, x2, possibleValue(1,j)));

        % Cross validate
        pred = svmPredict(model, Xval);
        error(i,j) = mean(abs(yval - pred));
    end
end

plot(error,'o');

% visualizeBoundary(X, y, model);