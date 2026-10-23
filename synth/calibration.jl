# Author: Noe Blassel
# Last modified: Oct 5th 2026

# gathers data for the risk calibration experiment (see Figures 9 and 10)
# usage (from repo root)   julia --project=. -t auto synth/calibration.jl

using FVDiagnostics
using Flux, JLD2, Random, Statistics
include(joinpath(@__DIR__, "config.jl"))

const NBATCH = 600
const ORDER = ["indist", "diff", "beta46", "beta46_1well", "beta46_diff"]

function main()
    model, input_dim, _ = load_diagnostic(joinpath(@__DIR__, "..", "models", "deployed.jld2"))
    testmode!(model)
    out = Dict{String, Any}()
    for name in ORDER
        βlims, potential_kwargs = variant(name)
        rng = Xoshiro(hash((:calibration, name)))
        p = Float64[]
        y = Float64[]
        traj = Int32[] # trajectory ids
        for _ in 1:NBATCH
            X, Y, mask = get_batch(rng; input_dim = input_dim, ncut = 0, ncorr = 2, βlims = βlims,
                                   potential_kwargs = potential_kwargs)
            P = Flux.σ(model(X))
            append!(p, Float64.(P[mask]))
            append!(y, Float64.(Y[mask]))
            n0 = isempty(traj) ? 0 : last(traj)
            for j in axes(mask, 2)
                append!(traj, fill(Int32(n0 + j), count(view(mask, :, j))))
            end
        end
        out[name] = (p = p, y = y, traj = traj, auc = rank_auc(p, y), base = mean(y)) # collect RNN predictions, decorrelation labels, ROC AUC and label balance
        println(name, ": ", length(p), " frames, positive rate ", round(mean(y); digits = 3), ", AUC ",
                round(out[name].auc; digits = 4))
    end
    jldsave(joinpath(DATA, "calibration.jld2"); conditions = out)
end

main()
