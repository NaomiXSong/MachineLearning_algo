function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

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
%
% Note: grad should have the same dimensions as theta
logPart(:,1) = log(sigmoid(X*theta));
logPart(:,2) = log(1-sigmoid(X*theta));% get the log part
yPart(:,1) =y;
yPart(:,2) = 1-y;%get the y part
combinePart = logPart.*yPart;%combine log and y
mPart = 1.0/m;
tmp =sum(combinePart(:,1))+sum(combinePart(:,2));% to calculate J
J = -mPart*tmp;

% Calculate the gradient
sumGrad1 = sum((sigmoid(X*theta)-y).*X(:,1));
grad(1,1) = mPart*sumGrad1;
sumGrad2 = (((sigmoid(X*theta)-y)'*X(:,2:end)))';
grad(2:end,1) = mPart*sumGrad2;

% =============================================================

end
