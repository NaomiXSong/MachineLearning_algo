function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

regTheta0 = theta(2:end);

tmp = (X * theta) - y;

reg = (lambda / (2*m)) * (regTheta0' * regTheta0);
nonreg = (1/(2*m)) * sum(tmp.^2);
J = nonreg + reg;

% Part 1.3
grad = (X' * tmp) / m;
grad(2:end)= grad(2:end)+ lambda * regTheta0 / m;










% =========================================================================

grad = grad(:);

end
