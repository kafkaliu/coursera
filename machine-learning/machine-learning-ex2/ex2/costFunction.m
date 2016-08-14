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
%
for i=1:m
  sig = sigmoid(X(i,:)*theta);
  J += -y(i)*log(sig)-(1-y(i))*log(1-sig);
endfor

J /= m;
% d=[sigmoid(X*theta),y]*[1;-1].*X; % the result of (hθ(x(i)) − y(i))x(i)
% calculate the sum of 1/m*(hθ(x(i)) − y(i))x(i)
grad = (ones(1,m)*([sigmoid(X*theta),y]*[1;-1].*X))'/m;

% =============================================================

end
