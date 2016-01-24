function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%

% size of Theta1 is 25 by 401
% size of Theta2 is 10 by 26
% size of X is 5000 by 400

% input layer
a1 = [ones(m,1) X]; %5000 by 401
z2 = a1 * Theta1'; % 5000 by 25

% hidden layer
a2 = sigmoid(z2); % 5000 by 25
a2 = [ones(m,1) a2]; % 5000 by 26

% result layer
z3 = a2 * Theta2'; % 5000 by 10

a3 = sigmoid(z3); % 5000 by 10

[Y I] = max(a3'); % Y will be the max probability and I will be the index of that max number

p = I';

% =========================================================================


end
