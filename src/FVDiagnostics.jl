# Author: Noe Blassel
# Last modified: Oct 5 2026

module FVDiagnostics

using Random, Statistics, LinearAlgebra, SparseArrays
using Arpack
using Flux, MLUtils
using JLD2, ProgressMeter

include("potential.jl")
include("features.jl")
include("generator.jl")
include("fleming_viot.jl")
include("batch.jl")
include("model.jl")
include("training.jl")
include("scoring.jl")

export generate_potential, comp_generator, comp_qsd, conv_tv, project_qsd, sim_fv
export get_bin, hist_feature, meta_scalar, tv_from_qsd
export draw_system, simulate_traces, get_batch
export CNNFeaturizer, RNNDiagnostic, RNNDiagnosticOnline, reset_rnn_state!, load_diagnostic
export TrainingRun, run_epoch!, test_accuracy!, masked_loss
export rank_auc, full_roc

end
