# Author: Noe Blassel
# Last modified: Oct 5th 2026

# Convenience script to print a summary of a model checkpoint to the terminal

# usage : (from repo root)  julia --project=. models/describe.jl [models/deployed.jld2]

using FVDiagnostics
using Flux

function main(path = isempty(ARGS) ? joinpath(@__DIR__, "deployed.jld2") : ARGS[1])
    model, input_dim, hp = load_diagnostic(path)
    println(path, ": ", hp)
    d = input_dim
    for layer in model.featurizer.encoder.layers
        layer isa Conv || continue
        cin, cout = size(layer.weight, 2), size(layer.weight, 3)
        d >>= 1
        println("  Conv(5, $cin => $cout) / MaxPool(2), leakyrelu   output $d x $cout   ",
                sum(length, Flux.trainables(layer)), " parameters")
    end
    for layer in model.rnn.layers
        println("  ", layer, "   ", sum(length, Flux.trainables(layer)), " parameters")
    end
    for layer in model.mlp_head.layers
        println("  ", layer, "   ", sum(length, Flux.trainables(layer)), " parameters")
    end
    println("total: ", sum(length, Flux.trainables(model)), " parameters")
end

main()
