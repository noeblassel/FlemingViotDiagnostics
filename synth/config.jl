# Author: Noe Blassel
# Last modified: Oct 5th 2026

# configurations of the synthetic experiments (Section 4.3)

const STRIDE_VALUES = [10, 25, 50, 100, 150, 200] # τ in number of time steps
const NREPLICAS_VALUES = [10, 25, 50, 100, 150, 200]
const NUM_BATCHES = 200 # batches used for scoring each (N,τ) cell
const NCORR = 10                                           # simulation horizon, in reference decorrelation times
const RSEED = 2027
const ALPHAS_NN = Float64.(Float32[0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.98, 0.99]) # detection thresholds for RNN
const ALPHAS_GR = [0.01, 0.02, 0.05, 0.1, 0.15, 0.2] # detection thresholds for Gelman-Rubin
const DETECT_BINS = 0.0:0.05:Float64(NCORR) # used for t_corr(α)/t_0 histograms

# name => (βlims, generate_potential keywords) for the prior shifts
const VARIANTS = Dict("indist" => ((1.0, 3.0), NamedTuple()),
                      "diff" => ((1.0, 3.0), (logσlims = (-1.0, 1.0),)),
                      "beta46" => ((4.0, 6.0), NamedTuple()),
                      "beta46_1well" => ((4.0, 6.0), (δlims = (0.0, 0.05),)),
                      "beta46_diff" => ((4.0, 6.0), (logσlims = (-1.0, 1.0),)))

const DATA = normpath(joinpath(@__DIR__, "..", "data", "synth")) # output data folder

"Grid cell (1:36) -> (τ index, N index), N fastest."
cell_from_index(k) = (div(k - 1, length(NREPLICAS_VALUES)) + 1, mod(k - 1, length(NREPLICAS_VALUES)) + 1)
cell_rng(si, ni) = Xoshiro(RSEED + 1_000_003 * si + 1_009 * ni) # per-cell rng state

function variant(name)
    haskey(VARIANTS, name) || throw(ArgumentError("unknown variant \"$name\", expected one of $(collect(keys(VARIANTS)))"))
    return VARIANTS[name]
end

function getarg(args, key, default)
    i = findfirst(a -> startswith(a, "--$key="), args)
    i === nothing && return default
    v = split(args[i], "="; limit = 2)[2]
    return default isa AbstractString ? String(v) : parse(typeof(default), v)
end
