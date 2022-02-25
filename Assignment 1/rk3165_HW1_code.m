X_train = table2array(readtable('X_train.csv'));%Reading x values of training dataset
Y_train = table2array(readtable('y_train.csv'));%Reading y values of training dataset
X_test = table2array(readtable('X_test.csv'));%Reading x values of test dataset
Y_test = table2array(readtable('y_test.csv'));%Reading y values of test dataset

%taking the SVD of the X values in dataset to use it in ridge regression
[U,S,V]=svd(X_train);

%After taking the SVD,Only the S matrix needs to be manipulated by adding
%the lambda term and we can estimate the w_rr parameters of the ridge
%regression.Also the degrees of freedom(df) is calculated for each value of
%lambda parallely in this loop itself whose formulae have been taken from
%the slides.
for lambda= 0:1:5000
    s_lambda_inv = S;
    df(lambda+1) =0;
    for i=1:7
        s_lambda_inv(i,i)=S(i,i)/(lambda+S(i,i)^2);
        df(lambda+1)= df(lambda+1)+S(i,i)^2/(lambda+S(i,i)^2);
    end
    w_rr(:,lambda+1) = V*transpose(s_lambda_inv)*transpose(U)*Y_train;
end

%Plot of various parameters' degree of freedom and its relation w.r.t to
%lambda 
figure(1)
plot(df,w_rr(1,:),'.-');
hold on
plot(df,w_rr(2,:),'.-');
plot(df,w_rr(3,:),'.-');
plot(df,w_rr(4,:),'.-');
plot(df,w_rr(5,:),'.-');
plot(df,w_rr(6,:),'.-');
plot(df,w_rr(7,:),'.-');

xlabel('Degrees of Freedom')
ylabel('w_{rr}')
title('Degrees of freedom plot for various w_{rr} parameters')
legend(' cylinders','displacement','horsepower','weight','acceleration','year made');
hold off

% Using the estimated paramter model to predict the new y values in the
% test dataset.
for i =1:1:101
    for j= 1:1:42 
        
       Y_predicted(i,j) = X_test(j,:)*w_rr(:,i);
       
    end
end

%calculation of RMSE for the above predicted values.
for i=1:1:101
    RMSE(i)= sqrt(sum((Y_test'-Y_predicted(i,:)).^2)/42);
    
end

%Plot of how RMSE varies for different values of lambda
figure(2)
plot(0:1:50,RMSE(1:51),'-x');
xlabel('lambda')
ylabel('RMSE')
title('Plot of RMSE error with different values of lambda')
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%
%PART-2 of the probelm.Polynomial Regression.

%Appending the x^2 terms in the matrix which is used in 2nd order
%polynomial regression.
X_train_2 =[X_train , X_train.^2]; 
X_train_2(:,7) =[];
X_test_2 = [X_test, X_test.^2];
X_test_2(:,7) =[];

%standardizing the x^2 terms in the dataset appended previously,with
%subtraction of mean and then dividing it by the standard deviation for
%each of the variables for the given dataset.
for i = 7:1:12
    t=X_train_2(:,i);
    mean1= mean(t);
    std1=std(t);
    X_train_2(:,i)= (X_train_2(:,i)-mean1)/std1;
    X_test_2(:,i)= (X_test_2(:,i)-mean1)/std1;
    
end

%Taking SVD for the ridge regression
[U2,S2,V2]=svd(X_train_2);

%Applying ridge regression formula as shown previously, to get the w_rr
%values.
for lambda= 0:1:100
    s_lambda_inv_2 = S2;
    for i=1:13
        s_lambda_inv_2(i,i)=S2(i,i)/(lambda+S2(i,i)^2);
    end
    w_rr_2(:,lambda+1) = V2*transpose(s_lambda_inv_2)*transpose(U2)*Y_train;
end

%Predicting the y values using the given values of w_rr using the test
%dataset.
for i =1:1:101
    for j= 1:1:42 
        
       Y_predicted_2(i,j) = X_test_2(j,:)*w_rr_2(:,i);
       
    end
end

%calculating RMSE for the 2nd order polynomial regression for various
%values of lambda
for i=1:1:101
    RMSE_2(i)= sqrt(sum((Y_test'-Y_predicted_2(i,:)).^2)/42);
end



%%%3rd order polynomial regression

%Appending the x^3 terms in the matrix which is used in 3rd order
%polynomial regression.
X_train_3 =[X_train_2,X_train.^3];
X_train_3(:,13) =[];
X_test_3 = [X_test_2, X_test.^3];
X_test_3(:,13) =[];

%standardizing the x^3 terms in the dataset appended previously,with
%subtraction of mean and then dividing it by the standard deviation for
%each of the variables for the given dataset.
for i = 13:1:18
    
    mean1= mean(X_train_3(:,i));
    std1=std(X_train_3(:,i));
    X_train_3(:,i)= (X_train_3(:,i)-mean1)/std1;
    X_test_3(:,i)= (X_test_3(:,i)-mean1)/std1;
    
end

%Taking SVD for the ridge regression
[U3,S3,V3]=svd(X_train_3);

%Applying ridge regression formula as shown previously, to get the w_rr
%values.
for lambda= 0:1:100
    s_lambda_inv_3 = S;
    for i=1:19
        s_lambda_inv_3(i,i)=S3(i,i)/(lambda+S3(i,i)^2);
    end
    w_rr_3(:,lambda+1) = V3*transpose(s_lambda_inv_3)*transpose(U3)*Y_train;
end

%Predicting the y values using the given values of w_rr using the test
%dataset.
for i =1:1:101
    for j= 1:1:42 
        
       Y_predicted_3(i,j) = X_test_3(j,:)*w_rr_3(:,i);
       
    end
end

%calculating RMSE for the 3rd order polynomial regression for various
%values of lambda
for i=1:1:101
    RMSE_3(i)= sqrt(sum((Y_test'-Y_predicted_3(i,:)).^2)/42);
end
%%
%plot of RMSE for various values of lambda for p=1,2 and 3 in polynomial
%regression
figure(3)
plot(0:1:100,RMSE,'-o');
hold on
plot(0:1:100,RMSE_2,'-.');
 plot(0:1:100,RMSE_3,'-x');
xlabel('lambda')
ylabel('RMSE')
title('Plot of RMSE error with different values of lambda')
legend('RMSE-p=1','RMSE-p=2','RMSE-p=3')
hold off





