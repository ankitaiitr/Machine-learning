function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
n=size(theta);
grad = zeros(size(theta));
for i=1:m, % Instructions: Compute the cost of a particular choice of theta.
    H=1/(1+exp(-theta'*X(i,:)'));
    J= J+ y(i)*log(H) +(1-y(i))*log(1-H);
    for j=1:size(grad)
        grad(j)=grad(j)+ (H-y(i))*X(i,j);
    end
end
theta_sq=theta.^2;
k=sum(theta_sq)-theta_sq(1);
J=-J/m+k*lambda/(2*m);
for j=1:size(grad),
if j==1,
grad(j)=grad(j)/m;
else
    grad(j)=grad(j)/m+lambda*theta(j)/m;
end;
end;

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta






% =============================================================

end
