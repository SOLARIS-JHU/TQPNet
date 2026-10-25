% Explicit MPC: offline solve + closed-loop simulation
% Records: offline_time, online_time, n_regions, memory, RMSE, cost
clear; clc;

% S1 (n=20, m=120) and S2 (n=45, m=270) only
files = {'mpc_n10.mat', 'mpc_n20.mat'};
N_QUERY = 100;
T_SIM   = 200;

results = {};
for fi = 1:length(files)
    d = load(files{fi});
    n = double(d.n_dec); m = double(d.m_con);
    ns = double(d.ns); nu = double(d.nu);
    np_dim = size(d.pF, 2);
    A_sys = d.A_sys; B_sys = d.B_sys;

    fprintf('\n=== %s: n=%d m=%d ns=%d nu=%d ===\n', files{fi}, n, m, ns, nu);

    r.name = files{fi};
    r.n = n; r.m = m; r.n_plus_m = n + m;
    r.ns = ns; r.nu = nu; r.np_dim = np_dim;

    try
        problem = Opt('H', d.H, 'f', d.f, 'pF', d.pF, ...
                      'A', d.Am, 'b', d.b, 'pB', d.pB, ...
                      'bndA', d.bndA, 'bndb', d.bndb);

        fprintf('  Solving explicit MPC...\n');
        t0 = tic;
        solution = problem.solve();
        offline_s = toc(t0);
        n_regions = solution.xopt.Num;

        % Memory (bytes)
        mem_params = n_regions * (n * np_dim + n) * 8;   % float64
        mem_poly   = n_regions * (m * np_dim) * 8;
        mem_total  = mem_params + mem_poly;

        fprintf('  Offline: %.2fs | Regions: %d | Memory: %.0f KB\n', ...
                offline_s, n_regions, mem_total/1024);

        % --- Online latency benchmark ---
        rng(999);
        P_test = -1 + 2*rand(np_dim, N_QUERY);
        solution.xopt.feval(P_test(:,1), 'primal'); % warmup

        qt = zeros(N_QUERY, 1);
        for i = 1:N_QUERY
            t1 = tic;
            solution.xopt.feval(P_test(:,i), 'primal');
            qt(i) = toc(t1) * 1000;
        end
        fprintf('  Online: median=%.4fms\n', median(qt));

        % --- Closed-loop simulation (same noise as Python: rng(123), U[-0.3,0.3]) ---
        rng(123);
        x = zeros(ns, 1);
        traj = zeros(T_SIM+1, ns);
        controls = zeros(T_SIM, nu);
        total_ms = 0;

        for t = 1:T_SIM
            t1 = tic;
            u_full = solution.xopt.feval(x, 'primal');
            total_ms = total_ms + toc(t1) * 1000;

            if isempty(u_full) || any(isnan(u_full))
                u = zeros(nu, 1);
            else
                u = u_full(1:nu);
            end
            u = max(-1, min(1, u));  % clip
            controls(t, :) = u';

            w = -0.3 + 0.6 * rand(ns, 1);
            x = A_sys * x + B_sys * u + w;
            traj(t+1, :) = x';
        end

        % Compute RMSE and cost (same formula as Python)
        x_states = traj(2:end, :);
        rmse = sqrt(mean(x_states(:).^2));
        cost = sum(x_states(:).^2) + sum(controls(:).^2);
        ms_per_step = total_ms / T_SIM;

        fprintf('  Closed-loop: RMSE=%.4f cost=%.1f ms/step=%.2f\n', ...
                rmse, cost, ms_per_step);

        r.offline_s = offline_s;
        r.n_regions = n_regions;
        r.mem_total_kb = mem_total / 1024;
        r.online_median_ms = median(qt);
        r.closedloop_ms = ms_per_step;
        r.rmse = rmse;
        r.cost = cost;
        r.status = 'success';

    catch e
        fprintf('  FAILED: %s\n', e.message);
        r.offline_s = -1; r.n_regions = -1; r.mem_total_kb = -1;
        r.online_median_ms = -1; r.closedloop_ms = -1;
        r.rmse = -1; r.cost = -1;
        r.status = 'failed';
    end
    results{end+1} = r;
end

% Save JSON
fid = fopen('explicit_mpc_results.json', 'w');
fprintf(fid, '[\n');
for i = 1:length(results)
    r = results{i};
    fprintf(fid, '  {"name":"%s","n":%d,"m":%d,"n_plus_m":%d,', r.name, r.n, r.m, r.n_plus_m);
    fprintf(fid, '"ns":%d,"nu":%d,', r.ns, r.nu);
    fprintf(fid, '"offline_s":%.4f,"n_regions":%d,"mem_total_kb":%.1f,', r.offline_s, r.n_regions, r.mem_total_kb);
    fprintf(fid, '"online_median_ms":%.6f,"closedloop_ms":%.6f,', r.online_median_ms, r.closedloop_ms);
    fprintf(fid, '"rmse":%.6f,"cost":%.4f,', r.rmse, r.cost);
    fprintf(fid, '"status":"%s"}', r.status);
    if i < length(results); fprintf(fid, ',\n'); else; fprintf(fid, '\n'); end
end
fprintf(fid, ']\n');
fclose(fid);
fprintf('\nDone. Saved explicit_mpc_results.json\n');
