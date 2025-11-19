% Testing infinity norm cost clustering
% Created by Microsoft Copilot
% Modified by Mikko Malinen, 2025

% Generate sample 2D data
%X = [randn(50,2)+2; randn(50,2)-2; randn(50,2)+[5 -3]];

X = load('s2.txt'); k = 15;
%X = load('../datasets/s2.txt'); k = 15;
%X = load('../datasets/birch1.txt'); k = 100;

% Cluster into k groups
[idx, centers] = kmeans_inf(X, k, 50);

% Plot results
figure; hold on;
gscatter(X(:,1), X(:,2), idx);
plot(centers(:,1), centers(:,2), 'kx', 'MarkerSize', 12, 'LineWidth', 2);
title('k-Means with L-infinity norm');
