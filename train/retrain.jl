# Author: Noe Blassel
# Last modified: Oct 5 2026

# Trains a model from a checkpointed model on new batches, with a new optimizer (Adam) state.
# usage : (from repo root)  julia --project=. -t auto train/retrain.jl [source.jld2] [key=value ...]

# Keyword arguments: lr, epochs, seed, train_batches, test_batches, pot_per_batch, trace_per_pot,
# cut_per_trace, stride_lo, stride_hi, Nreplicas_lo, Nreplicas_hi, beta_lo, beta_hi,
# val_every, output.

# (see parse_args and src/* for advanced usage)

using FVDiagnostics
using Flux, JLD2, Random, Statistics

const ROOT = joinpath(@__DIR__, "..")

function parse_args(args)
    pos = filter(a -> !contains(a, '='), args) # positional arguments
    kv = Dict(String(k) => String(v) for (k, v) in (split(a, '='; limit = 2) for a in args if contains(a, '='))) # keyword arguments
    get_val(k, default) = haskey(kv, k) ? (default isa String ? kv[k] : parse(typeof(default), kv[k])) : default
    source = isempty(pos) ? joinpath(ROOT, "models", "hpsearch_winner.jld2") : pos[1]
    return (source = source,
            lr = get_val("lr", 5e-4),
            epochs = get_val("epochs", 50),
            seed = get_val("seed", 2026),
            train_batches = get_val("train_batches", 50),
            test_batches = get_val("test_batches", 50),
            pot_per_batch = get_val("pot_per_batch", 5),
            trace_per_pot = get_val("trace_per_pot", 5),
            cut_per_trace = get_val("cut_per_trace", 2),
            stride_lims = (get_val("stride_lo", 10), get_val("stride_hi", 200)),
            Nreplicas_lims = (get_val("Nreplicas_lo", 10), get_val("Nreplicas_hi", 200)),
            betalims = (get_val("beta_lo", 1.0), get_val("beta_hi", 3.0)),
            val_every = get_val("val_every", 10),
            output = get_val("output", joinpath(ROOT, "models", "deployed.jld2")))
end

function main(args = ARGS)
    cfg = parse_args(args)
    model, input_dim, hp = load_diagnostic(cfg.source)
    println("source $(cfg.source), $(sum(length, Flux.trainables(model))) parameters, input_dim $input_dim")
    println(cfg)

    run = TrainingRun(rng = Xoshiro(cfg.seed), βlims = cfg.betalims, opt_state = Flux.setup(Adam(cfg.lr), model),
                      model = model, hp = hp, input_dim = input_dim, pot_per_batch = cfg.pot_per_batch,
                      trace_per_pot = cfg.trace_per_pot, cut_per_trace = cfg.cut_per_trace,
                      stride_lims = cfg.stride_lims, Nreplicas_lims = cfg.Nreplicas_lims)

    best_val_loss = Inf
    best_epoch = 0
    best_state = nothing
    for epoch in 1:cfg.epochs
        train_loss = mean(run_epoch!(run, cfg.train_batches))
        if epoch % cfg.val_every == 0 || epoch == cfg.epochs
            acc, val_loss = test_accuracy!(run, cfg.test_batches)
            if val_loss < best_val_loss
                best_val_loss, best_epoch, best_state = val_loss, epoch, deepcopy(Flux.state(model))
            end
            println("epoch $epoch  train $(round(train_loss; digits = 4))  val $(round(val_loss; digits = 4))  ",
                    "acc $(round(100acc; digits = 1))%")
        else
            println("epoch $epoch  train $(round(train_loss; digits = 4))")
        end
        flush(stdout)
    end

    run_config = Base.structdiff(cfg, NamedTuple{(:val_every, :output)})
    jldsave(cfg.output; model_state = best_state, hp = hp, lr = cfg.lr, best_epoch = best_epoch,
            best_val_loss = best_val_loss, run_config = run_config)
    println("best epoch $best_epoch, val $(round(best_val_loss; digits = 4)). Result saved to $(cfg.output)")
end

main()
