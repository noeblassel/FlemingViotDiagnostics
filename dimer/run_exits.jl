# Author: Noe Blassel
# Last modified: Nov 17 2026

# sample exit statistics from detected decorrelated Fleming-Viot populations.

# usage (from repo root):
#  julia --project=dimer -t auto dimer/run_exits.jl --setting=slow --mode=grid --cell=1 # from diagnosis frames in the (N,τ) cell with given index
#  julia --project=dimer -t auto dimer/run_exits.jl --setting=slow --mode=qsd --point=1 [--runs=1:48] # from from reference QSD samples associated with given initial condition

using JLD2, Random, Statistics
include(joinpath(@__DIR__, "config.jl"))
include(joinpath(@__DIR__, "fv.jl"))

"(id, seed, frames, column of the shard's xi) of each trajectory to re-run."
function exit_jobs(mode, g, args, setting_name)
    if mode == "grid"
        Dc = load(joinpath(dimer_setting_dir(setting_name), "detections.jld2"), "cells")[g["cell"]]
        return [(id = Dc.traj[j], seed = seed_grid(setting_name, g["cell"], Dc.traj[j]), frames = Dc.exit_frames[j],
                 col = Dc.traj[j]) for j in eachindex(Dc.traj)]
    else
        lo, hi = parse.(Int, split(getarg(args, "runs", "1:48"), ":"))
        return [(id = g["runs"][i], seed = seed_ref(setting_name, g["point"], g["runs"][i]), frames = [g["nframes"]], col = i)
                for i in lo:hi if g["ok"][i]]
    end
end

function main(args = ARGS)
    setting_name = getarg(args, "setting", "")
    a = setting(setting_name)
    dir = dimer_setting_dir(setting_name)
    mode = getarg(args, "mode", "grid")
    mode in ("grid", "qsd") || error("--mode must be grid or qsd")
    if mode == "grid"
        tag = getarg(args, "cell", 1)
        shard = joinpath(dir, "shards", "grid_c$(lpad(tag, 2, '0')).jld2")
        outf = joinpath(dir, "exits", "grid_c$(lpad(tag, 2, '0')).jld2")
    else
        tag = getarg(args, "point", 1)
        shard = joinpath(dir, "shards", "ref_p$(tag)_0001.jld2")
        outf = joinpath(dir, "exits", "qsd_p$tag.jld2")
    end

    g = jldopen(f -> Dict(k => f[k] for k in keys(f) if !(k in ("hist", "hist_nc", "ncoord", "gr", "gr_nc"))), shard, "r")
    dt, N, point = g["dt"], g["N"], g["point"]
    stride = mode == "grid" ? g["tau"] : g["stride"]
    q0 = load(joinpath(DATA, "dimer", "initial_conditions.jld2"), "coords")[point]

    λ_hat = mean(g["n_branch"]) / (N * g["nframes"] * stride * dt)  # empirical killing rate
    max_steps = round(Int, 25 / (λ_hat * dt))
    jobs = exit_jobs(mode, g, args, setting_name)
    println("$mode: point $(initial_config_labels[point]), d_dimer = $(a.d_dimer), N = $N, stride = $stride, ",
            "$(length(jobs)) trajectories, exit time cutoff $(round(max_steps * dt; digits = 1)) LJ")

    results = Vector{Any}(undef, length(jobs))
    t0 = time()
    Threads.@threads for jj in eachindex(jobs) # outer parallelizations - 
        jb = jobs[jj]
        fmax = maximum(jb.frames)
        r = run_fv(q0; N = N, stride = stride, nframes = fmax, dt = dt, d_dimer = a.d_dimer, seed = jb.seed,
                   nbins = nbins_max, record_hist = false, snapshot_frames = jb.frames)
        r.xi == g["xi"][:, 1:fmax, jb.col] || @warn "trajectory $(jb.id): re-run differs from the shard"
        results[jj] = [(traj = jb.id, frame = f, replica = rep,
                        exit = run_to_exit(r.snapshots[f][rep]; d_dimer = a.d_dimer, dt = dt,
                                           seed = seed_exit(setting_name, tag, jb.id, f, rep), max_steps = max_steps))
                       for f in jb.frames for rep in 1:N]
    end
    rows = reduce(vcat, results)
    println(length(rows), " exits (", count(r -> r.exit[2] == 0, rows), " censored) in ",
            round(time() - t0; digits = 0), " s")

    mkpath(dirname(outf))
    jldsave(outf; mode = mode, point = point, d_dimer = a.d_dimer, traj = [r.traj for r in rows],
            frame = [r.frame for r in rows], replica = [r.replica for r in rows],
            t_exit = [r.exit[1] for r in rows], side = [r.exit[2] for r in rows])
    println("wrote $outf")
end

main()
