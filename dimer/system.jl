# Author: Noe Blassel
# Last modified: Oct 28 2026

# Core functionality for the solvated dimer system (https://arxiv.org/abs/2610.36078, Section 4.4)

using Molly, StaticArrays, Random, LinearAlgebra
include(joinpath(@__DIR__, "potential.jl"))

## physical constants
const n_at_perdim= 8
const ρ_part = 0.693 # particle density
const n_atoms = n_at_perdim^3
const lattice_const = ρ_part^(-1 / 3)
const box_length = n_at_perdim* lattice_const
const wca_cutoff = 2^(1 / 6)
const temp = 1.0

## Molly constants

const atoms = [Atom(; index = i, mass = 1.0, σ = 1.0, ϵ = 1.0) for i in 1:n_atoms]

@inline dimer_r(coords, boundary) = norm(vector(coords[1], coords[2], boundary)) # periodic dimer length

const wca_pairs = let e = trues(n_atoms, n_atoms)
    e[1, 2] = e[2, 1] = false # dimer atoms do not interact through WCA
    e
end


## CV utilities

@inline bin_of(y, nbins) = 1 + clamp(floor(Int, nbins * y), 0, nbins - 1) # bin index in (0,1)

# soft coordination number
const softcoord_number_rc = wca_cutoff # "cutoff" radius
const softcoord_number_σ = 0.2 # "switching" width

function soft_ncoord(coords, boundary)
    n = 0.0
    @inbounds for j in 3:length(coords)
        d = min(sqrt(sum(abs2, vector(coords[1], coords[j], boundary))),
                sqrt(sum(abs2, vector(coords[2], coords[j], boundary))))
        n += 1 / (1 + exp((d - softcoord_number_rc) / softcoord_number_σ))
    end
    return n
end

@inline ξ(r) = (r - r_L) / width_Ω
@inline in_metastable_state(r) = r_L < r < r_R

## Molly functions
struct DimerInteraction end

@inline function Molly.force(::DimerInteraction, coord_i, coord_j, boundary, args...) # triple well force
    rij = vector(coord_i, coord_j, boundary)
    dij = norm(rij)
    f = dw_dr(dij) * (rij / dij)
    return SpecificForce2Atoms(f, -f)
end

@inline Molly.potential_energy(::DimerInteraction, coord_i, coord_j, boundary, args...) =
    w_of_r(norm(vector(coord_i, coord_j, boundary))) # triple well energy


lattice_coords() = reshape([SVector(i * lattice_const, j * lattice_const, k * lattice_const) for i in 0:(n_at_perdim- 1), j in 0:(n_at_perdim- 1),
                            k in 0:(n_at_perdim- 1)], n_atoms) # crystal configuration

function dimer_system(; coords = lattice_coords(), temp = 1.0, rng::AbstractRNG = Random.default_rng())
    boundary = CubicBoundary(box_length, box_length, box_length)
    wca = LennardJones(; cutoff = DistanceCutoff(wca_cutoff), use_neighbors = true)
    nf = CellListMapNeighborFinder(; eligible = wca_pairs, dist_cutoff = wca_cutoff,
                                   x0 = coords, unit_cell = boundary) # neighbor list for WCA pairs

    sys = System(; atoms = atoms,
                 coords = copy(coords),
                 velocities = zero(coords),
                 boundary = boundary,
                 pairwise_inters = (wca,),
                 specific_inter_lists = (InteractionList2Atoms([1], [2], [DimerInteraction()]),),
                 neighbor_finder = nf,
                 k = 1.0,
                 force_units = NoUnits,
                 energy_units = NoUnits)

    random_velocities!(sys, temp; rng = rng)
    return sys
end

# standard overdamped Langevin integrator - for equilibrium dynamics
overdamped(; dt = 1e-3, temp = 1.0, friction = 1.0) =
    OverdampedLangevin(; dt = dt, temperature = temp, friction = friction, remove_CM_motion = false)

"""
Euler-Maruyama integrator for Langevin dynamics with variable mobilities
"""
struct Integrator{F, B, N} # force function
    f::F # force buffer
    bufs::B # Molly buffers
    nbrs::Base.RefValue{N} # neighbor list
    dt::Float64 # timestep
    d_dimer::Float64 # dimer mobility coefficient
    σ::Float64
    σd::Float64
end

function Integrator(sys; dt, d_dimer)
    nbrs = find_neighbors(sys, sys.neighbor_finder; n_threads = 1)
    σ = sqrt(2 * dt * temp) # base diffusion coefficient
    return Integrator(Molly.zero_forces(sys), Molly.init_buffers!(sys, 1), Ref(nbrs), dt, d_dimer,
                      σ, σ * sqrt(d_dimer))
end

"""
Evolves a replica for one step with the integrator `it`, and returns ξ.
"""
function step!(sys, it::Integrator, rng, k)
    c = sys.coords
    Molly.forces!(it.f, sys, it.nbrs[], it.bufs, Val(false), k; n_threads = 1) # force computations are not parallelized for a single replica
    @inbounds for a in eachindex(c)
        m, s = a <= 2 ? (it.d_dimer, it.σd) : (1.0, it.σ)
        c[a] += m * it.dt * Molly.calc_accels(it.f[a], 1.0) +
                s * SVector(randn(rng), randn(rng), randn(rng))
        c[a] = wrap_coords(c[a], sys.boundary)
    end
    it.nbrs[] = find_neighbors(sys, sys.neighbor_finder, it.nbrs[], k; n_threads = 1)
    return ξ(dimer_r(c, sys.boundary))
end