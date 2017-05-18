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

logPart(:,1) = log(sigmoid(X*theta));
logPart(:,2) = log(1-sigmoid(X*theta));% get the log part
yPart(:,1) =y;
yPart(:,2) = 1-y;%get the y part
combinePart = logPart.*yPart;%combine log and y
mPart = 1.0/m;
tmp =sum(combinePart(:,1))+sum(combinePart(:,2));% to calculate J
J = -mPart*tmp + (lambda/(2*m))*sum(theta(2:end,1).^2);

% Calculate the gradient
sumGrad1 = sum((sigmoid(X*theta)-y).*X(:,1));
grad(1,1) = mPart*sumGrad1;
sumGrad2 = (((sigmoid(X*theta)-y)'*X(:,2:end)))';
grad(2:end,1) = mPart*sumGrad2+(lambda/m)*theta(2:end);




% =============================================================

end
