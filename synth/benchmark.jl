# Author: Noe Blassel
# Last modified: Oct 5th 2026

# runs one experiment on synthetic data = 1 combination of (N,τ,synthetic prior)
# usage (from repo root)   julia --project=. -t auto synth/benchmark.jl --variant=indist --cell=1 [--num_batches=200]
# valid keyword arguments
# --cell : an integer between 1 and 36, corresponding to a combination of (N,τ) (see config.jl)
# --variant : the prior, one of indist (training prior), diff (high D), beta46 (low T), beta46_1well (low T, one well) or beta46_diff (low T, high D)
# --num_batches : number of batches for evaluation (number of fleming-viot traces = 25 × num_batches)

using FVDiagnostics
using Flux, JLD2, Random
include(joinpath(@__DIR__, "config.jl"))

function score_cell(model, input_dim, k, βlims, potential_kwargs; num_batches)
    si, ni = cell_from_index(k)
    s, N = STRIDE_VALUES[si], NREPLICAS_VALUES[ni]
    rng = cell_rng(si, ni)
    rec = (tconv = Float32[], decorr = Int32[], len = Int32[], nn = Vector{Int32}[], gr = Vector{Int32}[],
           roc = NTuple{4, Float32}[])
    for _ in 1:num_batches
        X, Y, mask, B, tconv = get_batch(rng; input_dim = input_dim, stride_lims = (s, s), Nreplicas_lims = (N, N),
                                         ncut = 0, ncorr = NCORR, npot = 5, ntrace = 5, βlims = βlims,
                                         potential_kwargs = potential_kwargs)
        Flux.reset!(model)
        P = Flux.σ(model(X))
        for j in axes(Y, 2)
            len = sum(mask[:, j])
            len == 0 && continue
            decorr = findfirst(==(1.0f0), Y[:, j])
            decorr === nothing && continue
            push!(rec.tconv, tconv[j]) # reference decorrelation time
            push!(rec.decorr, decorr) # reference decorrelation frame
            push!(rec.len, len)

            # the classification task: pick a random frame below 2 decorrelation times
            # and record (RNN prediction, -(GR minus one), -(histogram TV), true label)
            # minus signs for GR and TV because lower statistics mean higher rank for these diagnostics
            k_roc = rand(rng, 1:min(len, 2decorr))
            push!(rec.roc, (P[k_roc, j], -B[1, k_roc, j], -B[2, k_roc, j], Y[k_roc, j]))

            # record decorrelation times for GR and RNN, for the threshold values in config.jl
            push!(rec.nn, [something(findfirst(>(α), view(P, 1:len, j)), 0) for α in ALPHAS_NN])
            push!(rec.gr, [something(findfirst(<(α), view(B, 1, 1:len, j)), 0) for α in ALPHAS_GR])
        end
    end
    return (si = si, ni = ni, log = rec)
end

"""
runs the experiment and saves to a JLD2 the results for this cell and prior variant
"""
function main(args = ARGS)
    name = getarg(args, "variant", "indist")
    βlims, potential_kwargs = variant(name)
    k = getarg(args, "cell", 1)
    1 <= k <= length(STRIDE_VALUES) * length(NREPLICAS_VALUES) || error("cell $k out of range")
    num_batches = getarg(args, "num_batches", NUM_BATCHES)
    model, input_dim, _ = load_diagnostic(joinpath(@__DIR__, "..", "models", "deployed.jld2"))
    testmode!(model)
    si, ni = cell_from_index(k)
    println("$name, cell $k: tau = $(STRIDE_VALUES[si]), N = $(NREPLICAS_VALUES[ni]), $num_batches batches")
    t0 = time()
    r = score_cell(model, input_dim, k, βlims, potential_kwargs; num_batches = num_batches)
    out = joinpath(DATA, "shards", name, "cell_$(lpad(k, 2, '0')).jld2")
    mkpath(dirname(out))
    jldsave(out; r..., num_batches = num_batches)
    println(length(r.log.len), " trajectories in ", round(time() - t0; digits = 0), " saved to $out")
end

main()
