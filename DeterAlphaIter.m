function Alfa = DeterAlphaIter( A,y,S1,S2,alpha)
%% Determine the Regularization Parameter for Adaptive regularization
%  Author: Kunpu Ji 
%  Date: 2021/07/18
%  Input: A -- coefficient matrix (m x n)
%         y -- observation vector (m x 1)
%         S1 -- the index sets for the terms which are not worthy of regularization
%         S2 -- the index sets for the terms which should be regularized by Tikhonov method 
%         alpha -- the regularization parameter of the former iteration
%  Output: Alfa -- the updated regularization parameter
Nmat = A'*A;
[U,S,V] = svd(A);
[m,n] = size(A);
S3 = setdiff(setdiff(1:n,S1),S2);
lambda = diag(S(1:n,1:n));
lambda_S1 = 1./lambda(S1);
lambda_S2 = lambda(S2)./(lambda(S2).^2+alpha);
xu = (V(:,S1)*diag(lambda_S1)*U(:,S1)'+V(:,S2)*diag(lambda_S2)*U(:,S2)')*y;
eu = y - A*xu;
sigma2 = eu'*eu - sum(alpha^2*(lambda(S2).^2./(lambda(S2).^2+alpha).^2).*(V(:,S2)'*xu).^2) - sum(lambda(S3).^2.*(V(:,S3)'*xu).^2);
sigma2 = sigma2/(m - length(S1) - length(S2) + alpha^2*sum(1./(lambda(S2).^2 + alpha).^2));
Alfa = RegParm(Nmat,sqrt(sigma2),xu,S2);
end

