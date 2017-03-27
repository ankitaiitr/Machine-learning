function p = predict(theta, X)
%PREDICT Predict whether the label is 0 or 1 using learned logistic 
%regression parameters theta
%   p = PREDICT(theta, X) computes the predictions for X using a 
%   threshold at 0.5 (i.e., if sigmoid(theta'*x) >= 0.5, predict 1)

m = size(X, 1); % Number of training examples

% You need to return the following variables correctly
p = zeros(m, 1);

k=sigmoid(X*theta) % ====================== YOUR CODE HERE ======================
for i=1:m,% Instructions: Complete the following code to make predictions using
if k(i)>=0.5,
p(i)=1;
else%               your learned logistic regression parameters. 
p(i)=0;%               You should set p to a vector of 0's and 1's
end%







% =========================================================================


end
