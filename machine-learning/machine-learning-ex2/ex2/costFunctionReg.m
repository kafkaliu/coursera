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

for i=1:m
  sig = sigmoid(X(i,:)*theta);
  J += -y(i)*log(sig)-(1-y(i))*log(1-sig);
endfor

J = J/m + lambda/(2*m)*sum(theta(2:length(theta)).^2);
% d=[sigmoid(X*theta),y]*[1;-1].*X; % the result of (hθ(x(i)) − y(i))x(i)
% calculate the sum of 1/m*(hθ(x(i)) − y(i))x(i)
grad = (ones(1,m)*([sigmoid(X*theta),y]*[1;-1].*X))'/m + [0;lambda*theta(2:length(theta))/m];

% =============================================================

end
