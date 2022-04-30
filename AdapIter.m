function xo = AdapIter( A,y,alpha_initial,delta1,delta2,maxiter)
%% Iterative Adaptive Regularization
%  Author: Kunpu Ji 
%  Date: 2021/07/18
%  Input: A -- coefficient matrix (m x n)
%         y -- observation vector (m x 1)
%         alpha_initial -- given the initial regularization parameter
%         delta1 -- the small value for terminating the iteration in 
%                   computing the regularization parameter using minimum MSE criteria
%         delta2 -- the small value for terminating the iteration in
%                   computing the adaptive regularized solution
%         maxiter -- the maximum iteration number
%  Output: xo -- Adaptive reguarized solution (n x 1)
[m,n] = size(A);
[U,S,V] = svd(A);
lambda = diag(S(1:n,1:n));
N = A'*A;
alpha = DeterAlpha( A,y,alpha_initial,delta1);
alpha0 = alpha;
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
xu = (V(:,S1)*diag(lambda_S1)*U(:,S1)'+V(:,S2)*diag(lambda_S2)*U(:,S2)')*y; 
eu = y - A*xu;
S3 = setdiff(setdiff(ZZ',S1),S2);
sigma2 = eu'*eu - sum(alpha^2*(lambda(S2).^2./(lambda(S2).^2+alpha).^2).*(V(:,S2)'*xu).^2) - sum(lambda(S3).^2.*(V(:,S3)'*xu).^2);
sigma2 = sigma2/(m - length(S1) - length(S2) + alpha^2*sum(1./(lambda(S2).^2 + alpha).^2));
lambda_alpha2 = lambda(S2)./(lambda(S2).^2+alpha);
lambda_b2 = alpha./(lambda(S2).^2+alpha);
bXr = -(V(:,S3)*V(:,S3)'+V(:,S2)*diag(lambda_b2)*V(:,S2)')*xu;
DXr = sigma2*(V(:,S1)*diag(1./(lambda(S1).^2))*V(:,S1)'+V(:,S2)*diag(lambda_alpha2.^2)*V(:,S2)');
mse_r = trace(DXr+ bXr*bXr');
err_old = mse_r;
alpha_O = alpha;
xo = xu;
DXr2u = sigma2*(V(:,S1)*diag(1./(lambda(S1).^2))*V(:,S1)'+V(:,S2)*diag(lambda_alpha2.^2)*V(:,S2)');
bXr2u = -(V(:,S3)*V(:,S3)'+V(:,S2)*diag(lambda_b2)*V(:,S2)')*xu;
cnt = 1;
while 1
    if cnt > maxiter
        break;
    end
    S3 = setdiff(setdiff(1:n,S1),S2);
    if isempty(S2)
        alpha = 0;
    else
       [ alpha ] = DeterAlphaIter( A,y,S1,S2,alpha);
    end
    lambda_S1 = 1./lambda(S1);
    lambda_S2 = lambda(S2)./(lambda(S2).^2+alpha);
    xu = (V(:,S1)*diag(lambda_S1)*U(:,S1)'+V(:,S2)*diag(lambda_S2)*U(:,S2)')*y;
    eu = y - A*xu;
    sigma2 = eu'*eu - sum(alpha^2*(lambda(S2).^2./(lambda(S2).^2+alpha).^2).*(V(:,S2)'*xu).^2) - sum(lambda(S3).^2.*(V(:,S3)'*xu).^2);
    sigma2 = sigma2/(m - length(S1) - length(S2) + alpha^2*sum(1./(lambda(S2).^2 + alpha).^2));
    lambda_alpha2 = lambda(S2)./(lambda(S2).^2+alpha);
    lambda_b2 = alpha./(lambda(S2).^2+alpha);
    bXr2u = -(V(:,S3)*V(:,S3)'+V(:,S2)*diag(lambda_b2)*V(:,S2)')*xu;
    DXr2u = sigma2*(V(:,S1)*diag(1./(lambda(S1).^2))*V(:,S1)'+V(:,S2)*diag(lambda_alpha2.^2)*V(:,S2)');
    MXr2u = DXr2u+ bXr2u*bXr2u';
    err_Ji = trace(MXr2u);
    if  (err_Ji > err_old || abs(err_Ji-err_old)<delta2) 
        if alpha_O>alpha0
            [ alpha ] = mse( alpha0/3, alpha0,A,y,100,sigma);
            con_S1 = sigma2*(2/alpha+1./(lambda.^2))-(V'*xu).^2;
            con_S2 = lambda.^2 - sigma2./((V'*xu).^2) + 2*alpha;
            [~, S1] = find(con_S1' < 0);
            [~, S2] = find(con_S2' > 0);
            S2 = intersect(setdiff(1:n,S1),S2);
            lambda_S1 = 1./lambda(S1);
            lambda_S2 = lambda(S2)./(lambda(S2).^2+alpha);
            xu = (V(:,S1)*diag(lambda_S1)*U(:,S1)'+V(:,S2)*diag(lambda_S2)*U(:,S2)')*y;
            xo = xu;
            alpha_O = alpha;
            %break;
        else
            break;
        end
    else
        con_S1 = sigma2*(2/alpha+1./(lambda.^2))-(V'*xu).^2;
        con_S2 = lambda.^2 - sigma2./((V'*xu).^2) + 2*alpha;
        [~, S1] = find(con_S1' < 0);
        [~, S2] = find(con_S2' > 0);
        S2 = intersect(setdiff(1:n,S1),S2);
        xo = xu;
        err_old = err_Ji;
        alpha_O = alpha; 
    end
    cnt = cnt + 1;
end
end

