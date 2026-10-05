# Author: Noe Blassel
# Last modified: Oct 5th 2026

# Processes results for one prior specification (see config.jl) from its 36 cell shards and computes
# - per cell ROC-AUC scores (for RNN, GR and histogram TV)
# - total ROC-AUC scores
# - full ROC curves
# - decorrelation time histograms (relative to reference, for RNN and GR) for each detection threshold (see config.jl)
# - additional metadata (see below)

# Saves the processed results in JSON (data/synth/{variant}_benchmark_results.json)

# Additionally, aggregates per-trajectory data (produced by benchmark.jl) into a single JLD2 file (data/synth/{variant}_trajectories.jld2)

# usage (from repo root)   julia --project=. -t auto synth/merge.jl --variant=indist
# valid keyword arguments
# --variant : the prior, one of indist (training prior), diff (high D), beta46 (low T), beta46_1well (low T, one well) or beta46_diff (low T, high D)

using FVDiagnostics
using JLD2, JSON
include(joinpath(@__DIR__, "config.jl"))

"""
relative decorrelation time histogram, used in Figure 8
"""
function detect_hist(frames, decorr, nalpha)
    nb = length(DETECT_BINS) - 1
    H = zeros(Int, nb, nalpha)
    for (fr, d) in zip(frames, decorr), (ai, k) in enumerate(fr)
        k > 0 && (H[get_bin(k / d, first(DETECT_BINS), last(DETECT_BINS), nb), ai] += 1)
    end
    return H
end

function main(args = ARGS)
    name = getarg(args, "variant", "indist")
    dir = joinpath(DATA, "shards", name)
    ns, nn = length(STRIDE_VALUES), length(NREPLICAS_VALUES)
    files = [joinpath(dir, "cell_$(lpad(k, 2, '0')).jld2") for k in 1:(ns * nn)]
    missing_cells = findall(!isfile, files)
    isempty(missing_cells) || error("missing cells $missing_cells in $dir")
    shards = [load(f) for f in files]
    length(unique(s["num_batches"] for s in shards)) == 1 || error("shards with different num_batches")


    # concatenate results from each shard
    si = reduce(vcat, [fill(s["si"], length(s["log"].len)) for s in shards])
    ni = reduce(vcat, [fill(s["ni"], length(s["log"].len)) for s in shards])
    cat_log(f) = reduce(vcat, [getfield(s["log"], f) for s in shards])
    decorr, roc = cat_log(:decorr), cat_log(:roc)
    nn_frames, gr_frames = cat_log(:nn), cat_log(:gr)
    label = [r[4] for r in roc] # one roc sample is = (rnn output, -gr output, -tv ouput, label)  (see benchmark.jl)

    J = Dict{String, Any}("stride_values" => STRIDE_VALUES, "Nreplicas_values" => NREPLICAS_VALUES,
                          "detect_bins" => collect(DETECT_BINS), "alpha_values" => ALPHAS_NN,
                          "gr_alpha_values" => ALPHAS_GR,
                          "traj_count_grid" => [count((si .== a) .& (ni .== b)) for a in 1:ns, b in 1:nn],
                          "detect_hist" => detect_hist(nn_frames, decorr, length(ALPHAS_NN)),
                          "gr_detect_hist" => detect_hist(gr_frames, decorr, length(ALPHAS_GR)))
    for (m, p) in ((1, ""), (2, "gr_"), (3, "tv_"))
        score = [r[m] for r in roc]
        J["$(p)rank_auc"] = rank_auc(score, label)
        J["$(p)rank_auc_grid"] = [(c = (si .== a) .& (ni .== b); rank_auc(score[c], label[c])) for a in 1:ns, b in 1:nn]
        froc = full_roc(score, label)
        J["$(p)roc_fpr_full"], J["$(p)roc_tpr_full"] = froc.fpr, froc.tpr # full ROC curve
    end
    out = joinpath(DATA, "$(name)_benchmark_results.json")
    open(io -> write(io, JSON.json(J; allownan = true)), out, "w")

    traj_log = (si = Int16.(si), ni = Int16.(ni), tconv = cat_log(:tconv), decorr = decorr, len = cat_log(:len),
                detect = [reduce(hcat, nn_frames), reduce(hcat, gr_frames)])
    jldsave(joinpath(DATA, "$(name)_trajectories.jld2"); traj_log = traj_log, stride_values = STRIDE_VALUES,
            Nreplicas_values = NREPLICAS_VALUES, dt = 1e-3, methods = ["", "gr_"], alpha_values = [ALPHAS_NN, ALPHAS_GR])
    println("rank AUC: NN ", round(J["rank_auc"]; digits = 4), ", GR ", round(J["gr_rank_auc"]; digits = 4),
            ", TV ", round(J["tv_rank_auc"]; digits = 4), "  ($(length(decorr)) trajectories) -> $out")
end

main()
