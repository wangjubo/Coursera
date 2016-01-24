function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

[m n] = size(X);
sum = 0;
for i = 1:m
    z = theta' * X(i,:)';
    h = sigmoid(z);
    sum = sum + (-y(i) * log(h) - (1 - y(i)) * log(1 - h));
end
J_unreg = sum / m;

sum = 0;
for j = 2:n
    sum = sum + theta(j) ^ 2;
end
J_reg = sum * lambda / 2 / m;

J = J_unreg + J_reg;

%Calculate gradient
for j = 1:n
    sum = 0;
    for i = 1:m
        z = theta' * X(i,:)';
        h = sigmoid(z);
        sum = sum + (h - y(i)) * X(i,j);
    end
    grad(j,1) = sum / m;
    
    if(j > 1)
        grad(j,1) = grad(j,1) + lambda * theta(j) / m;
    end
end

% =============================================================

end
