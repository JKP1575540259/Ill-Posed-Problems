function alpha = DeterAlpha( A,y,alpha_initial,delta1)
%% Determine the Regularization Parameter with minimum MSE criteria
%  Author: Kunpu Ji 
%  Date: 2021/07/18
%  Input:  A -- coefficient matrix (m x n)
%          y -- observation vector (m x 1)
%          alpha_initial -- given the initial regularization parameter
%          delta1 -- the smaller value for terminating the iteration in 
%                    computing the regularization parameter using minimum MSE criteria
%  Output: the determined regularization parameter
alpha_old = alpha_initial;
[m,n] = size(A);
Nmat = A' * A;
alpha_new = alpha_old;
cnt = 1;
while 1
    Qalpha = inv(Nmat + alpha_new * eye(n));
    Xalpha = Qalpha * (A' * y);
    ealpha = A * Xalpha - y;
    sigma2 = (ealpha' * ealpha - alpha_new^2 * Xalpha' * (Qalpha - alpha_new * Qalpha * Qalpha) * Xalpha) / (m - n + alpha_new^2 * trace(Qalpha * Qalpha));
    alpha_new = RegParm(Nmat,sqrt(sigma2),Xalpha,1:n);
    if abs(alpha_new - alpha_old) < delta1 || cnt > 100
        break;
    else
        alpha_old = alpha_new;
    end
    cnt = cnt + 1;
end
alpha = alpha_new;
end




