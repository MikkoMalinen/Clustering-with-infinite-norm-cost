function [idx, centers] = kmeans_inf(X, k, maxIter)
% KMEANS_INF clusters data using k-means style algorithm with L-infinity norm
%
% Inputs:
%   X        - (n x d) data matrix, n points in d dimensions
%   k        - number of clusters
%   maxIter  - maximum number of iterations
%
% Outputs:
%   idx      - cluster assignment for each point (n x 1)
%   centers  - cluster centers (k x d)
%
%   Created with Microsoft Copilot
%   Made available by Mikko Malinen, 2025

    if nargin < 3
        maxIter = 100;
    end
    
    [n, d] = size(X);
    
    % Random initialization of centers
    rng('shuffle');
    centers = X(randperm(n, k), :);
    
    idx = zeros(n,1);
    
    for iter = 1:maxIter
        % --- Assignment step ---
        for i = 1:n
            % Compute L-infinity distance to each center
            dists = max(abs(centers - X(i,:)), [], 2);
            [~, idx(i)] = min(dists);
        end
        
        % --- Update step ---
        newCenters = zeros(k, d);
        for j = 1:k
            clusterPoints = X(idx == j, :);
            if ~isempty(clusterPoints)
                % Use component-wise median (robust for L-infinity)
                newCenters(j,:) = median(clusterPoints, 1);
            else
                % If cluster is empty, reinitialize randomly
                newCenters(j,:) = X(randi(n), :);
            end
        end
        
        % Check for convergence
        if all(all(abs(newCenters - centers) < 1e-6))
            break;
        end
        
        centers = newCenters;
    end
end
