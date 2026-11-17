# Author: Noe Blassel
# Last modified: Nov 17 2026

# compute diagnostic traces from fleming-viot trajectories of the dimer system, in a given cell index

# usage (from repo root):
#   julia --project=dimer -t auto dimer/run_grid.jl --setting=slow --cell=1 [--out=path]

using JLD2, Random, Statistics
include(joinpath(@__DIR__, "config.jl"))
include(joinpath(@__DIR__, "fv.jl"))

function main(args = ARGS)
    setting_name = getarg(args, "setting", "")
    a = setting(setting_name)
    cell = getarg(args, "cell", 1)
    1 <= cell <= n_cells() || error("cell $cell outside 1:$(n_cells())")
    outf = getarg(args, "out", joinpath(dimer_setting_dir(setting_name), "shards", "grid_c$(lpad(cell, 2, '0')).jld2"))

    ti, ni, point = cell_to_idx(cell)
    stride, N = τ_grid[ti], n_grid[ni]
    ref = load(joinpath(dimer_setting_dir(setting_name), "reference.jld2"), "points")[point]
    nframes = ceil(Int, n_corr_dimer * max(ref.t_corr_ref, ref.joint.t_corr_ref) / (stride * sim_dt))
    q0 = load(joinpath(DATA, "dimer", "initial_conditions.jld2"), "coords")[point]
    println("cell $cell: tau = $stride, N = $N, point $(initial_config_labels[point]), $n_fv_trajectories trajectories x $nframes frames, ",
            "d_dimer = $(a.d_dimer), $(Threads.nthreads()) threads")

    hist_xi = Array{Int32}(undef, nbins_max, nframes, n_fv_trajectories)
    hist_nc = Array{Int32}(undef, nbins_max, nframes, n_fv_trajectories)
    xi = Array{Float32}(undef, N, nframes, n_fv_trajectories)
    n_coord = Array{Float32}(undef, N, nframes, n_fv_trajectories)
    gr_xi = Array{Float64}(undef, nframes, n_fv_trajectories)
    gr_nc = Array{Float64}(undef, nframes, n_fv_trajectories)
    n_branchings = zeros(Int, n_fv_trajectories)
    fail = fill("", n_fv_trajectories)

    t0 = time()
    Threads.@threads for j in 1:n_fv_trajectories # parallelize over serial fleming-viot simulations (no communication between fv trajectories)
        try
            r = run_fv(q0; N = N, stride = stride, nframes = nframes, dt = sim_dt, d_dimer = a.d_dimer,
                       seed = seed_grid(setting_name, cell, j), nbins = nbins_max, nc_range = a.nc_range)
            hist_xi[:, :, j] .= r.hist
            hist_nc[:, :, j] .= r.hist_nc
            xi[:, :, j] .= r.xi
            n_coord[:, :, j] .= r.ncoord
            gr_xi[:, j] .= r.gr
            gr_nc[:, j] .= r.gr_nc
            n_branchings[j] = r.n_branch
        catch e
            fail[j] = sprint(showerror, e)
            xi[:, :, j] .= NaN32
        end
    end
    println(count(isempty, fail), "/$n_fv_trajectories trajectories ok in ", round(time() - t0; digits = 1), " s")
    foreach(j -> isempty(fail[j]) || println("  trajectory $j failed: ", first(fail[j], 120)), 1:n_fv_trajectories)

    mkpath(dirname(outf))
    jldsave(outf; cell = cell, tau = stride, N = N, point = point, label = initial_config_labels[point], r0 = initial_dimer_lengths[point],
            meta_scalar = sqrt(N * stride * sim_dt), nframes = nframes, dt = sim_dt, d_dimer = a.d_dimer,
            nbins = nbins_max, nc_range = a.nc_range, ok = isempty.(fail), hist = hist_xi, hist_nc = hist_nc, xi = xi,
            ncoord = n_coord, gr = gr_xi, gr_nc = gr_nc, n_branch = n_branchings, failures = fail)
    println("wrote $outf")
end

main()
