struct CNNFeaturizer{S <: Chain}
    encoder::S
    output_dim::Int
end

Flux.@layer CNNFeaturizer

"""
    CNNFeaturizer(; input_dim, kernel_dims, nchannels, rng = Random.GLOBAL_RNG)

The featurizing layer of the RNN model.

Several layers of 1D convolutions/leaky ReLU/max-pooling of width 2, on an `input_dim`-size input feature (histograms)
Layers are specified by compatible vectors of `kernel_dims` and `nchannels`.
"""
function CNNFeaturizer(; input_dim::Int, kernel_dims::Vector{Int}, nchannels::Vector{Int},
                       rng = Random.GLOBAL_RNG)
    initializer = Flux.glorot_uniform(rng)
    cnn_layers = []
    in_channels = 1

    for (kernel_size, out_channels) in zip(kernel_dims, nchannels)
        pad_size = div(kernel_size - 1, 2)
        push!(cnn_layers, Conv(tuple(kernel_size), in_channels => out_channels, leakyrelu; pad = pad_size,
                               init = initializer))
        push!(cnn_layers, MaxPool((2,)))
        in_channels = out_channels
    end

    encoder = Chain(cnn_layers...)
    output_dim = prod(Flux.outputsize(encoder, (input_dim, 1, 1)))
    return CNNFeaturizer(encoder, output_dim)
end

(f::CNNFeaturizer)(x) = reshape(f.encoder(x), f.output_dim, size(x, 3))   # (input_dim, 1, batch size) → (f.output_dim, batch size)

struct RNNDiagnostic{F <: CNNFeaturizer, T <: Chain, U <: Chain}
    featurizer::F
    rnn::T
    mlp_head::U
end

Flux.@layer RNNDiagnostic

"""
    RNNDiagnostic(featurizer; dims_rnn = [64], dims_mlp = [64, 32], rng = Random.GLOBAL_RNG)

The RNN model, consisting of
- A `featurizer` (for instance `CNNFeaturizer`)
- A LSTM recurrent neural network, specified by a vector of latent dimensions `dims_rnn`
- A dense feedforward classifier, specified by a vector of hidden dimensions `dims_mlp`
"""
function RNNDiagnostic(featurizer::CNNFeaturizer;
        dims_rnn::Vector{Int} = [64],
        dims_mlp::Vector{Int} = [64, 32],
        rng = Random.GLOBAL_RNG)
    initializer = Flux.glorot_uniform(rng)

    rnn_layers = []
    input_dim_rnn = featurizer.output_dim + 1
    for output_dim_rnn in dims_rnn
        push!(rnn_layers, LSTM(input_dim_rnn => output_dim_rnn; init_kernel = initializer,
                               init_recurrent_kernel = initializer))
        input_dim_rnn = output_dim_rnn
    end
    rnn = Chain(rnn_layers...)

    mlp_layers = []
    input_dim_mlp = last(dims_rnn)
    for output_dim_mlp in dims_mlp
        push!(mlp_layers, Dense(input_dim_mlp => output_dim_mlp, leakyrelu; init = initializer))
        input_dim_mlp = output_dim_mlp
    end
    mlp = Chain(mlp_layers..., Dense(last(dims_mlp) => 1; init = initializer))

    return RNNDiagnostic(featurizer, rnn, mlp)
end

"""
    RNNDiagnostic(hp; rng = Random.GLOBAL_RNG)

RNNDiagnostic specified by a hyperparameter `hp`
    (featurizer = (input_dim_exponent, depth, width_exponent),
     rnn_depth, rnn_width_exponent, mlp_depth, mlp_width_exponent)

These hyperparameters specify the following architecture.

- CNN architecture
    - Input dimension: `2^input_dim_exponent` (number of histogram bins)
    - Number of CNN layers = `depth`, kernel size = 5, nchannels = `[2^width_exponent,..., 2^(width_exponent + depth - 1)]`.
- LSTM architecture
    - `rnn_depth` layers with latent size `2^rnn_width_exponent`
- Classifier architecture
    - `mlp_depth` hidden layers with hidden size `2^mlp_width_exponent`

"""
function RNNDiagnostic(hp::NamedTuple; rng = Random.GLOBAL_RNG)
    f = hp.featurizer
    featurizer = CNNFeaturizer(; input_dim = 2^f.input_dim_exponent, kernel_dims = fill(5, f.depth),
                               nchannels = [2^i for i in f.width_exponent:(f.width_exponent + f.depth - 1)],
                               rng = rng)
    dims_rnn = fill(2^hp.rnn_width_exponent, hp.rnn_depth)
    dims_mlp = fill(2^hp.mlp_width_exponent, hp.mlp_depth)
    return RNNDiagnostic(featurizer; dims_rnn = dims_rnn, dims_mlp = dims_mlp, rng = rng)
end

function (m::RNNDiagnostic)(x)
    ndims(x) == 3 || throw(DimensionMismatch("expected (input_dim + 1, length, batch), got $(size(x))"))

    total_dim, seq_len, batch_size = size(x)
    feature_dim = total_dim - 1
    x_feat = x[1:feature_dim, :, :] # input feature
    x_meta = x[(feature_dim + 1):end, :, :] # design feature

    z = m.featurizer(reshape(x_feat, feature_dim, 1, seq_len * batch_size)) # flatten featurizer output
    z = vcat(reshape(z, size(z, 1), seq_len, batch_size), x_meta) # concatenate with design feature

    yhat = m.mlp_head(m.rnn(z))                            # predict logits (1, length, batch)
    return reshape(yhat, seq_len, batch_size)
end

mutable struct RNNDiagnosticOnline{F <: CNNFeaturizer, T <: AbstractVector, U <: Chain, V}
    featurizer::F
    rnn_cells::T
    mlp_head::U
    rnn_state::V
end

Flux.@layer RNNDiagnosticOnline

"""
    RNNDiagnosticOnline(model::RNNDiagnostic)

Deployment form of the RNN diagnostic: the model is called one frame at the time, storing the LSTM state between
calls. Internal state is reset with `reset_rnn_state!`.
"""
function RNNDiagnosticOnline(model::RNNDiagnostic)
    rnn_cells = [layer.cell for layer in model.rnn.layers]
    dims_rnn = [size(cell.bias, 1) ÷ 4 for cell in rnn_cells]
    rnn_state = [(zeros(Float32, d), zeros(Float32, d)) for d in dims_rnn]
    return RNNDiagnosticOnline(model.featurizer, rnn_cells, model.mlp_head, rnn_state)
end

reset_rnn_state!(model::RNNDiagnosticOnline) =
    (model.rnn_state = [(zero(h), zero(c)) for (h, c) in model.rnn_state])

function (m::RNNDiagnosticOnline)(x)
    ndims(x) == 1 || throw(DimensionMismatch("expected one frame of length input_dim + 1, got $(size(x))"))
    feature_dim = length(x) - 1
    z = vcat(vec(m.featurizer(reshape(x[1:feature_dim], feature_dim, 1, 1))), x[(feature_dim + 1):end])

    for (i, cell) in enumerate(m.rnn_cells)
        (z, m.rnn_state[i]) = cell(z, m.rnn_state[i])
    end

    return m.mlp_head(z)
end

"""
    load_diagnostic(path) -> (model, input_dim, hp)

Rebuilds a `RNNDiagnostic` from a checkpoint file `path`. Returns the reconstructed model, its input dimension, and its hyperparameters
"""
function load_diagnostic(path::AbstractString)
    ckpt = JLD2.load(path)
    hp = ckpt["hp"] # hyperparameter
    model = RNNDiagnostic(hp; rng = Xoshiro(0)) # rebuild architecture
    Flux.loadmodel!(model, ckpt["model_state"]) # saved weights
    return model, 2^hp.featurizer.input_dim_exponent, hp # model, input_dim, other hyperparameters
end
