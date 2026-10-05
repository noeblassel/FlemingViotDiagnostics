"""
    TrainingRun(; rng, βlims, opt_state, model, hp, input_dim, pot_per_batch, trace_per_pot,
                cut_per_trace, stride_lims, Nreplicas_lims)
- `rng` : pseudo-random number generator
- `opt_state` : optimizer state
- `model` : the model undergoing training
- `hp` : model hyperparameters
- `input_dims`,βlims,`pot_per_batch`,`trace_per_pot`,`cut_per_trace`,`stride_lims`,`N_replicas_lims`: see `get_batch` in batch.jl
"""
@kwdef mutable struct TrainingRun{R, S, T, U, H}
    rng::R
    βlims::S
    opt_state::T
    model::U
    hp::H
    input_dim::Int
    pot_per_batch::Int
    trace_per_pot::Int
    cut_per_trace::Int
    stride_lims::Tuple{Int, Int}
    Nreplicas_lims::Tuple{Int, Int}
end

_batch(run::TrainingRun) = get_batch(run.rng;
                                     input_dim = run.input_dim,
                                     ntrace = run.trace_per_pot,
                                     npot = run.pot_per_batch,
                                     ncut = run.cut_per_trace,
                                     βlims = run.βlims,
                                     stride_lims = run.stride_lims,
                                     Nreplicas_lims = run.Nreplicas_lims)

masked_loss(logits, Y, mask) = Flux.logitbinarycrossentropy(logits, Y; agg = x -> mean(x[mask]))

"""
    test_accuracy!(run, n) -> (acc, loss)

Average accuracy over `n` fresh batches and valid frames within each batch, for the threshold α=0.5
"""
function test_accuracy!(run::TrainingRun, n)
    acc = 0.0
    loss = 0.0
    testmode!(run.model)

    @showprogress for _ in 1:n
        X, Y, mask = _batch(run)
        logits = run.model(X)
        acc += mean((Flux.σ(logits)[mask] .> 0.5) .== Y[mask])
        loss += masked_loss(logits, Y, mask)
    end

    return acc / n, loss / n
end

"""
    run_epoch!(run, n) -> losses

Runs optimization on `n` training batches (with parameters specified by `run`); returns the training loss at each step.
"""
function run_epoch!(run::TrainingRun, n)
    trainmode!(run.model)
    losses = Float32[]

    @showprogress for _ in 1:n
        X, Y, mask = _batch(run)
        loss, grads = Flux.withgradient(m -> masked_loss(m(X), Y, mask), run.model)
        Flux.update!(run.opt_state, run.model, grads[1])
        push!(losses, loss)
    end

    return losses
end
