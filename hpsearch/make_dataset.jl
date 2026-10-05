# Author: Noe Blassel
# Last modified: October 5 2026

# Generates the dataset for the hyperparameter search and saves it to $REPO_ROOT/data/hpsearch/hp_dataset.jld2
# 1000 Fleming–Viot traces
# 200 samples from training priors, 5 trajectories per sample
# 800 / 200 training and evaluation split

# usage: (from repo root)  julia --project=hpsearch -t auto hpsearch/make_dataset.jl

using FVDiagnostics
using JLD2, Random, Statistics, ProgressMeter

const DATA = normpath(joinpath(@__DIR__, "..", "data", "hpsearch"))

function generate_offline_dataset(rng; npot = 200, ntrace = 5, ncorr = 3, tol = 0.05, Ngrid = 100,
                                  βlims = (1.0, 3.0), dt = 1e-3, tau_gt = 0.1, stride_lims = (10, 200),
                                  Nreplicas_lims = (10, 200), min_length = 5, max_attempts = 10)
    traces = []
    i = 0
    prog = Progress(npot)
    while i < npot
        sys = draw_system(rng; βlims, Ngrid, tau_gt)
        rs = simulate_traces(rng, sys; ntrace, stride_lims, Nreplicas_lims, Ngrid, dt, tol, tau_gt,
                             min_length, ncorr, max_attempts)
        any(r -> !r.success, rs) && continue # discard failed attempts
        append!(traces, rs)
        i += 1
        next!(prog)
    end
    return (sequences = [r.fv_frames for r in traces],
            labels = [Float32.((1:length(r.fv_frames)) .* (r.stride * dt) .> r.T_conv) for r in traces],
            Nreplicas = [r.Nreplicas for r in traces], strides = [r.stride for r in traces],
            T_convs = [r.T_conv for r in traces], dt = dt)
end

subset(ds, idx) = (sequences = ds.sequences[idx], labels = ds.labels[idx], Nreplicas = ds.Nreplicas[idx],
                   strides = ds.strides[idx], T_convs = ds.T_convs[idx], dt = ds.dt)

function main()
    ds = generate_offline_dataset(Xoshiro(2026))
    n = length(ds.sequences)
    perm = randperm(Xoshiro(2026), n)
    ntrain = round(Int, 0.8n)
    mkpath(DATA)
    jldsave(joinpath(DATA, "hp_dataset.jld2"); train = subset(ds, perm[1:ntrain]), test = subset(ds, perm[(ntrain + 1):end]))
    println("$n traces -> ", joinpath(DATA, "hp_dataset.jld2"))
end

main()
