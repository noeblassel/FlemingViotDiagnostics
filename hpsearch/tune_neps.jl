## Author: Blassel
## Last modified: Oct 5th 2026

# Uses the Neural Pipeline Search (NePS) to optimize architecture (and learning-rate) hyperparameters
# Algorithm: in-context freeze-thaw BO (IfBO)
# automl.github.io/neps/latest/reference/search_algorithms/multifidelity/#3-in-context-freeze-thaw-bayesian-optimization

# usage (from repo root)   julia --project=hpsearch -t auto hpsearch/tune_neps.jl

using FVDiagnostics
using Flux, JLD2, Statistics, Random, MLUtils, ProgressMeter, PythonCall
include(joinpath(@__DIR__, "offline.jl")) # batching logic for offline dataset


const DATA = normpath(joinpath(@__DIR__, "..", "data", "hpsearch"))
const RESULTS_DIR = joinpath(DATA, "neps_results")
const BASE_SEED = 2025
const NPARAMS_MAX = 500_000
const NPARAMS_MIN = 10_000
const CHECKPOINT_NAME = "checkpoint.jld2"

const HP_TRAIN = JLD2.load(joinpath(DATA, "hp_dataset.jld2"), "train")
const HP_TEST = JLD2.load(joinpath(DATA, "hp_dataset.jld2"), "test")
const neps = pyimport("neps")

const SPACE = (learning_rate = neps.Float(; lower = 1e-4, upper = 1e-2, log = true),
               rnn_depth = neps.Integer(; lower = 1, upper = 2),
               rnn_width_exp = neps.Integer(; lower = 4, upper = 6),
               mlp_depth = neps.Integer(; lower = 1, upper = 2),
               mlp_width_exp = neps.Integer(; lower = 4, upper = 6),
               cnn_depth = neps.Integer(; lower = 3, upper = 5),
               cnn_width_exp = neps.Integer(; lower = 3, upper = 4),
               input_dim_exp = neps.Integer(; lower = 5, upper = 8),
               epoch = neps.IntegerFidelity(; lower = 1, upper = 30))
# The hyperparameter search space.
# The number of training epochs is a fidelity parameter, limited to 30 per hyperparameter configuration


# `depth = min(c.cnn_depth, c.input_dim_exp)` because each CNN layer halves the input dimension via max pooling
hp_of(c) = (featurizer = (input_dim_exponent = c.input_dim_exp, depth = min(c.cnn_depth, c.input_dim_exp),
                          width_exponent = c.cnn_width_exp),
            rnn_depth = c.rnn_depth, rnn_width_exponent = c.rnn_width_exp,
            mlp_depth = c.mlp_depth, mlp_width_exponent = c.mlp_width_exp)

function build_run(lr, hp, seed)
    rng = Xoshiro(seed)
    model = RNNDiagnostic(hp; rng = rng)
    return TrainingRun(rng = rng, βlims = (1.0, 3.0), opt_state = Flux.setup(Adam(lr), model), model = model,
                       hp = hp, input_dim = 2^hp.featurizer.input_dim_exponent, pot_per_batch = 5,
                       trace_per_pot = 5, cut_per_trace = 2, stride_lims = (10, 200), Nreplicas_lims = (10, 200))
end

"""
The objective function optimized by NePS.
"""
function evaluate_pipeline(; pipeline_directory, previous_pipeline_directory, epoch, kwargs...)
    pipe_dir = string(pipeline_directory)                  # pathlib.Path -> String
    prev_dir = pyis(previous_pipeline_directory, pybuiltins.None) ? nothing : string(previous_pipeline_directory)
    epochs = pyconvert(Int, epoch)
    c = (; learning_rate = pyconvert(Float64, kwargs[:learning_rate]),
         (k => pyconvert(Int, kwargs[k]) for k in keys(SPACE) if k ∉ (:learning_rate, :epoch))...)
    hp = hp_of(c)
    run = build_run(c.learning_rate, hp, BASE_SEED + hash(c))
    np = sum(length, Flux.trainables(run.model))
    println("epoch $epochs, $np parameters, lr $(round(c.learning_rate; sigdigits = 3))")
    (NPARAMS_MIN <= np <= NPARAMS_MAX) || return pydict(Dict("objective_to_minimize" => 1.0, "cost" => 1))
    # short-circuit training if model falls outside of acceptable parameter range.
    # The training loss is always <= 1.0, so `objective_to_minimize` is an upper bound

    epochs_done = 0
    if prev_dir !== nothing && isfile(joinpath(prev_dir, CHECKPOINT_NAME))
        ckpt = JLD2.load(joinpath(prev_dir, CHECKPOINT_NAME)) # thaws a training run
        Flux.loadmodel!(run.model, ckpt["model_state"])
        run.opt_state = Flux.setup(Adam(c.learning_rate), run.model)
        epochs_done = ckpt["epoch"]
    end
    for e in (epochs_done + 1):epochs
        println("  epoch $e/$epochs  train loss ", round(run_epoch_offline!(run, HP_TRAIN); digits = 4))
    end
    result = test_loss_offline!(run, HP_TEST)
    println("  test loss ", round(result.loss; digits = 4), ", accuracy ", round(100result.acc; digits = 1), "%")

    mkpath(pipe_dir)
    jldsave(joinpath(pipe_dir, CHECKPOINT_NAME); model_state = Flux.state(run.model), epoch = epochs, hp = hp) # freezes a training run
    return pydict(Dict("objective_to_minimize" => result.loss, "cost" => epochs - epochs_done))
end

# first wrapper : jl_fn is a Python wrapper for `evaluate_pipeline`
const _ns = pydict(Dict("jl_fn" => pyfunc(evaluate_pipeline))) 

# second wrapper: neps_evaluate wraps around jl_fn with an explicit signature (this one is fed to NePS)
pybuiltins.exec(Py("def neps_evaluate(pipeline_directory, previous_pipeline_directory, " * join(keys(SPACE), ", ") *
                   "):\n    return jl_fn(**locals())"), _ns)

# run the optimizer
neps.run(; evaluate_pipeline = _ns["neps_evaluate"],
         pipeline_space = pybuiltins.type("HPOSpace", pytuple((neps.PipelineSpace,)),
                                          pydict(Dict(String(k) => v for (k, v) in pairs(SPACE))))(),
         root_directory = RESULTS_DIR, optimizer = "ifbo", fidelities_to_spend = 3000)

println("Hyperparameter search done, results saved to $RESULTS_DIR.")