# Author: Noe Blassel
# Last modified: Oct 5th 2026

# Utilities to extract batches from offline dataset

const OFFLINE_MINIBATCH_SIZE = 32
const PAD = -1.0f0

"""
    assemble_batch(dataset, indices, rng; input_dim, min_length = 5)

`(X, Y, mask)` of the traces in `dataset` indexed by `indices`,
cut uniformly at random between its decorrelation frame and its end (as in `get_batch`).

Ignores traces shorter than `min_length`.
"""
function assemble_batch(dataset, indices, rng; input_dim, min_length = 5)
    dt = dataset.dt
    batch = Vector{Vector{Float32}}[]
    labels = Vector{Float32}[]
    for k in indices
        frames = dataset.sequences[k]
        l = length(frames)
        l < min_length && continue
        meta = meta_scalar(dataset.Nreplicas[k], dataset.strides[k], dt)
        features = [vcat(hist_feature(f, input_dim), [meta]) for f in frames]
        decorr_frame = max(min_length, ceil(Int, dataset.T_convs[k] / (dataset.strides[k] * dt)))
        len = clamp(rand(rng, decorr_frame:l), min_length, l)
        push!(batch, features[1:len])
        push!(labels, Float32.(dataset.labels[k][1:len]))
    end
    isempty(batch) && return nothing
    Y = stack(MLUtils.batchseq(labels, PAD); dims = 1)
    return stack(MLUtils.batchseq(batch, zeros(Float32, input_dim + 1)); dims = 2), Y, (Y .!= PAD)
end

"Trains `run` over one epoch (that is, one permutation of `dataset), returning the mean training loss."
function run_epoch_offline!(run::TrainingRun, dataset; minibatch_size::Int = OFFLINE_MINIBATCH_SIZE)
    trainmode!(run.model)
    losses = Float32[]
    @showprogress for chunk in Iterators.partition(randperm(run.rng, length(dataset.sequences)), minibatch_size)
        mb = assemble_batch(dataset, chunk, run.rng; input_dim = run.input_dim)
        mb === nothing && continue
        X, Y, mask = mb
        loss, grads = Flux.withgradient(m -> masked_loss(m(X), Y, mask), run.model)
        Flux.update!(run.opt_state, run.model, grads[1])
        push!(losses, Float32(loss))
    end
    return isempty(losses) ? 0.0f0 : mean(losses)
end

"Computes the average loss and accuracy (at decision threshold α=0.5) over `dataset`."
function test_loss_offline!(run::TrainingRun, dataset; minibatch_size::Int = OFFLINE_MINIBATCH_SIZE)
    testmode!(run.model)
    losses = Float64[]
    n_correct = 0
    n_total = 0
    @showprogress for chunk in Iterators.partition(randperm(run.rng, length(dataset.sequences)), minibatch_size)
        mb = assemble_batch(dataset, chunk, run.rng; input_dim = run.input_dim)
        mb === nothing && continue
        X, Y, mask = mb
        logits = run.model(X)
        push!(losses, Float64(masked_loss(logits, Y, mask)))
        n_correct += sum((Flux.σ.(logits)[mask] .> 0.5f0) .== Y[mask])
        n_total += sum(mask)
    end
    return (loss = isempty(losses) ? 0.0 : mean(losses), acc = n_total > 0 ? n_correct / n_total : 0.0)
end
