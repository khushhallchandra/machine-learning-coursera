function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%
c_values=[0.01; 0.03; 0.1; 0.3; 1; 3; 8; 20];
sigma_values=[0.01; 0.03; 0.1; 0.3; 1; 3; 10; 30];

for i=1:length(c_values)
    for j=1:length(sigma_values)
        model= svmTrain(X, y, c_values(i), @(x1, x2) gaussianKernel(x1, x2, sigma_values(j))); 
        predictions = svmPredict(model,Xval);
        error_j(j) = mean(double(predictions ~= yval));
    end
    error_i(:,i) = error_j';    
end

[row,col]=find(min(min(error_i)) == error_i);
C=c_values(col);
sigma=sigma_values(row);
% =========================================================================

end
