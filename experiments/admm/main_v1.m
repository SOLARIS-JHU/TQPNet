clear; clc; close all;

% min 0.5 x^T Q x + d^T x, s.t. A x <= b

n = 500;
m = 500;
numQP = 100;
maxIter = 10000;
rho = 1.0;

A_scale = 4;

Res_ADMM3_all = zeros(maxIter, numQP);
Res_ADMM2_all = zeros(maxIter, numQP);

opts = optimoptions('quadprog', ...
    'Display', 'off', ...
    'Algorithm', 'interior-point-convex');

for k = 1:numQP
    fprintf('Solving random QP %d / %d...\n', k, numQP);

    %% Generate random QP
    Q = randn(n,n);
    Q = Q' * Q + 1e-6 * eye(n);   % small regularization
    d = 2 * rand(n,1) - 1;

    A = A_scale*sprand(m,n,0.1);
    b = A_scale*rand(m,1);

    %% Reference solution
    x_opt = quadprog(Q, d, A, b, [], [], [], [], [], opts);

    %% Standard 3-block ADMM iteration
    x = zeros(n,1);
    z = zeros(m,1);
    y = zeros(m,1);

    K3 = Q + rho * (A' * A);
    K3_inv = inv(K3);

    for iter = 1:maxIter
        x = K3_inv * (-d + A' * (z - y));
        z = min(b, A*x + y);
        y = y + A*x - z;

        Res_ADMM3_all(iter,k) = norm(x - x_opt);
    end

    %% Low-dimensional 2-block ADMM iteration
    [i_idx, j_idx, a_val] = find(A);

    DU = accumarray(i_idx, a_val.^2, [m 1], @sum, 0);
    DU = max(DU, 1e-12);
    invDU = 1 ./ DU;

    DV = accumarray(j_idx, 1, [n 1], @sum, 0);
    DV = max(DV, 1e-12);

    K2 = Q + rho * diag(DV);
    K2_inv = inv(K2);
    x = zeros(n,1);
    beta = max(0, -invDU .* b);

    for iter = 1:maxIter
        x_next = K2_inv * (rho * (DV .* x) - rho * A' * beta - d);

        beta = max(0, beta ...
            + invDU .* (A * (2*x_next - x)) ...
            - invDU .* b);

        x = x_next;

        Res_ADMM2_all(iter,k) = norm(x - x_opt);
    end
end

%% Compute statistics
q_ADMM3_05 = prctile(Res_ADMM3_all, 5, 2);
q_ADMM3_50 = prctile(Res_ADMM3_all, 50, 2);
q_ADMM3_95 = prctile(Res_ADMM3_all, 95, 2);

q_ADMM2_05 = prctile(Res_ADMM2_all, 5, 2);
q_ADMM2_50 = prctile(Res_ADMM2_all, 50, 2);
q_ADMM2_95 = prctile(Res_ADMM2_all, 95, 2);

iters = (1:maxIter)';

%% Plot median + 5%-95% percentile band
figure; hold on; box on;

fill([iters; flipud(iters)], ...
     [q_ADMM3_05; flipud(q_ADMM3_95)], ...
     'r', ...
     'FaceAlpha', 0.18, ...
     'EdgeColor', 'none');

fill([iters; flipud(iters)], ...
     [q_ADMM2_05; flipud(q_ADMM2_95)], ...
     'b', ...
     'FaceAlpha', 0.18, ...
     'EdgeColor', 'none');

semilogy(iters, q_ADMM3_50, 'r-', 'LineWidth', 2.0);
semilogy(iters, q_ADMM2_50, 'b-', 'LineWidth', 2.0);

xlabel('Iteration');
ylabel('$\|x^k - x^\star\|_2$', 'Interpreter', 'latex');
% title('Convergence Statistics over 100 Random QPs');

legend({'Full-dimensional ADMM 5%-95%', ...
        'Low-dimensional ADMM 5%-95%', ...
        'Full-dimensional ADMM median', ...
        'Low-dimensional ADMM median'}, ...
        'Location', 'best');

grid on;
set(gca, 'YScale', 'log');
set(gca, ...
    'LineWidth', 1.2, ...
    'FontSize', 10, ...
    'Box', 'on'); 