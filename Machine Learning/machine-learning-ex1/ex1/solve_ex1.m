clc;
close all;

data=load('ex1data1.txt');
theta0 = 0;
theta1 = 0;
alpha = 0.003;
x = data(:,1);
y = data(:,2);
iteration_num = 100;
cost_arr = zeros(iteration_num,1);
[m, n] = size(data);
figure(1)
plot(x, y, 'o'); hold on;
for i=1:iteration_num
    diff_theta0 = 0;
    diff_theta1 = 0;
    for j=1:m
        diff_theta0 = diff_theta0 + theta0 + theta1 * x(j) - y(j);
        diff_theta1 = diff_theta1 + (theta0 + theta1 * x(j) - y(j)) * x(j);
    end
    theta0 = theta0 - diff_theta0 * alpha / m;
    theta1 = theta1 - diff_theta1 * alpha / m;
    
    figure(1)
    plot(x, theta0 + theta1 * x);
    hold on;
    
    % Compute the cost function
    cost = 0;
    for j=1:m
        cost = cost + (theta0 + theta1 * x(j) - y(j))^2 / 2 / m;
    end
    cost_arr(i,1) = cost;
    
end

figure(2);
plot(cost_arr)
