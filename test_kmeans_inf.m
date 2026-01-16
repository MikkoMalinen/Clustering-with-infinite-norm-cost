% Testing infinity norm cost clustering
% Created by Microsoft Copilot
% Modified by Mikko Malinen, 2025

clear all;

tic;

REPEATS = 10;
Best_cost_L_infty = Inf;

% Generate sample 2D data
%X = [randn(50,2)+2; randn(50,2)-2; randn(50,2)+[5 -3]];

%X = load('s2.txt'); k = 15;
%X = load('../datasets/s2.txt'); k = 15;
%X = load('../datasets/birch1.txt'); k = 100;
X = load('../datasets/iris.txt'); k = 3;

n = size(X,1);
d = size(X,2);

for repeats = 1:REPEATS

% Cluster into k groups
[idx, centers] = kmeans_inf(X, k, 50);

% Cost

Cost_L_infty = 0;
for i = 1:n
Dist(i,:) = max(max(abs(X(i,:)-centers(idx(i),:))));
Cost_L_infty = Cost_L_infty + Dist(i,:);
end
if Cost_L_infty < Best_cost_L_infty
    Best_cost_L_infty = Cost_L_infty;
    idx_best = idx;
    centers_best = centers;
end

end % repeats

toc;

sz = zeros(k,1);  % sizes of clusters
for i = 1:n
    sz(idx_best(i)) = sz(idx_best(i))+1;
    CM(idx_best(i),sz(idx_best(i)),1:d) = X(i,1:d);  % CM has points of clusters
end
for j = 1:k  
    clear CM_2;
    CM_2(1:sz(j),1:2) = CM(j,1:sz(j),1:2);
    clear idx_clust;
    [idx_clust, vol] = convhulln(CM_2);
if j==1
    idx_b_clust1 = idx_clust(:,1);
elseif j==2
    idx_b_clust2 = idx_clust(:,1);
elseif j==3
    idx_b_clust3 = idx_clust(:,1);
end

end


% Plot results
figure; hold on;
gscatter(X(:,1), X(:,2), idx);
plot(centers(:,1), centers(:,2), 'kx', 'MarkerSize', 12, 'LineWidth', 2);
title('k-Means with L-infinity norm');
for j=1:k   %1:k
    hold on;
    clear CH;
    clear CM_2;
    CM_2(1:sz(j),1:2) = CM(j,1:sz(j),1:2);
    if j==1
        CH(:,1:2) = CM_2(idx_b_clust1,1:2);
    elseif j==2
        CH(:,1:2) = CM_2(idx_b_clust2,1:2);
    elseif j==3
        CH(:,1:2) = CM_2(idx_b_clust3,1:2);
    end
    plot(CH(:,1),CH(:,2),'k');
    CH_last(1,:) = CH(end,1:2);
    CH_last(2,:) = CH(1,1:2);
    hold on;
    plot(CH_last(:,1),CH_last(:,2),'k');
end

Best_cost_L_infty = Best_cost_L_infty 