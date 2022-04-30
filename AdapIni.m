function Xu = AdapIni( A,y,alpha_initial,delta1)
%% Initial Adaptive Regularization
%  Author: Kunpu Ji 
%  Date: 2021/07/18
%  Input: A -- coefficient matrix (m x n)
%         y -- observation vector (m x 1)
%         alpha_initial -- given the initial regularization parameter
%         delta1 -- the smaller value for terminating the iteration in 
%                   computing the regularization parameter using minimum MSE criteria
%  Output: Xu -- Adaptive reguarized solution (n x 1)
[m,n] = size(A);
[U,S,V] = svd(A);
lambda = diag(S(1:n,1:n));
N = A'*A;
alpha = DeterAlpha( A,y,alpha_initial,delta1 );
Qalpha = inv(N + alpha * eye(n));
Xr = Qalpha * (A' * y);
ealpha = A * Xr - y;
sigma2 = (ealpha' * ealpha - alpha^2 * Xr' * (Qalpha - alpha * Qalpha * Qalpha) * Xr) / (m - n + alpha^2 * trace(Qalpha * Qalpha));
[S1,~] = find((sigma2*(2/alpha+1./lambda.^2) -(V'*Xr).^2 )< 0);
con_S2 = lambda.^2 - sigma2./((V'*Xr).^2) + 2*alpha;
[S2,~] = find(con_S2 > 0);
ZZ = 1:n;
S2 = intersect(setdiff(ZZ',S1),S2);
lambda_S1 = 1./lambda(S1);
lambda_S2 = lambda(S2)./(lambda(S2).^2+alpha);
Xu = (V(:,S1)*diag(lambda_S1)*U(:,S1)'+V(:,S2)*diag(lambda_S2)*U(:,S2)')*y; 
end