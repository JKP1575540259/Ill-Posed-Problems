function Alfa = RegParm(Nmat,Sigma,DeltU,index_22)
% This function is used to compute the regularization parameter 
% By Shen Yunzhong June,14,2007
Kn = size(Nmat);
Knum = Kn(1);
% InvN = inv(Nmat);
SigmaSq = Sigma*Sigma;
% The result of basic ridge
dUTdU = DeltU'*DeltU;
Kk = Knum*SigmaSq/dUTdU;
clear InvN;
[Un,Sn,Un]=svd(Nmat);
Sn = diag(Sn);
Yn = Un'*DeltU;
K1 = 0;
ii = 0;
% Determine the first value of Kk which makes the Fk>0
while 1
   ii = ii+1;
   Fk = 0;
   for i =1:length(index_22)
      Fk = Fk+Sn(index_22(i))*(Kk*Yn(index_22(i))^2-SigmaSq)/(Kk+Sn(index_22(i)))^3;
   end
   if Fk > 0 
      K2 = Kk;
      break;
   else
      Kk = 1.5*Kk;
   end
end
jj = 0;
while 1
   jj = jj+1;
   Fk = 0;
   for i = 1:length(index_22)
      Fk = Fk+Sn(index_22(i))*(Kk*Yn(index_22(i))^2-SigmaSq)/(Kk+Sn(index_22(i)))^3;
   end
   if Fk > 0
      K2 = Kk;
   else
      K1 = Kk;
   end
   
   if abs(K2-K1)<K2*1e-8
      break;
   end
   Kk = K1+(K2-K1)/2;
end
   
Alfa = Kk;


% Computing the bias corrected regularization parameter
%% ii = 0;
%% K1 = Kk;
%% while 1
%%   ii = ii+1;
%%   Fk = 0;
%%  for i =1:Knum
%%      Fk = Fk+Sn(i)*(Kk^2*Yn(i)^2-SigmaSq*(Sn(i)+2*Kk))/(Kk+Sn(i))^5;
%%  end
%%  if Fk > 0 
%%      K2 = Kk;
%%      break;
%%   else
%%      Kk = 1.2*Kk;
%%   end
%% end
%%jj = 0;

%% while 1
%%    jj = jj+1;
%%    Fk = 0;
%%    for i = 1:Knum
%%       Fk = Fk+Sn(i)*(Kk^2*Yn(i)^2-SigmaSq*(Sn(i)+2*Kk))/(Kk+Sn(i))^5;
%%    end
   
%%    if Fk > 0
%%       K2 = Kk;
%%    else
%%       K1 = Kk;
%%    end
   
%%    if abs(K2-K1)<K2*1e-8
%%       break;
%%    end
%%    Kk = K1+(K2-K1)/2;
%% end
 
%% Alf_Bc = Kk;

