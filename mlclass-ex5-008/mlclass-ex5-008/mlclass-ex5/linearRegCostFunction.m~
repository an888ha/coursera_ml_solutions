function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

t = size(X);
size_g = t(2);

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%


theta_t = theta;
theta_t(1) = 0;
J =   (sum((X*theta - y).^2))/(2*m) + (sum(theta_t.^2)*lambda)/(2*m);  

temp1 = sum( (X*theta - y).*X(:,1));
grad(1) = 1/m*(temp1);
i = 2;
while(i<=size_g)
	temp2 = sum( (X*theta - y).*X(:,i));
	grad(i) = 1/m*(temp2) + (lambda/m)*theta(i);
	i = i + 1;
	end










% =========================================================================

grad = grad(:);

end
