"""
    generate_potential(; n_points = 1000, clims = (0.15, 0.85), δlims = (0.0, 0.2),
                       κlims = (0.2, 0.6), logσlims = (-4, -1), mmax = 6,
                       poneWell = 0.0, pconstD = 0.5, rng = Random.default_rng())

Samples a potential `V` and a diffusion profile `D` on (0, 1) from the prior distribution, specified by
- `clims`, `κlims`: centre and width ranges for the Gaussian main well
- `δlims`: range for the internal barrier heights, and for the boundary energy difference
- `logσlims`: range of logarithmic scales for the diffusion profile `D`
- `mmax`: maximum number of perturbations in `V` and in `D`
- `poneWell`: probability of no internal barriers in `V`
- `pconstD`: probability of no perturbation in `D`

The parameter `n_points` is used to sample min/max values for normalization of `V` and `D`
Returns a tuple of closures `(V, D, V′, D′)`

"""
function generate_potential(;
        n_points::Int = 1000,
        clims = (0.15, 0.85),
        δlims = (0.0, 0.2),
        κlims = (0.2, 0.6),
        logσlims = (-4, -1),
        mmax = 6,
        poneWell = 0.0,
        pconstD = 0.5,
        rng::AbstractRNG = Random.default_rng())
    δmin, δmax = δlims
    cmin, cmax = clims
    c_mw = cmin + (cmax - cmin) * rand(rng)
    κmin, κmax = κlims
    κ = κmin + rand(rng) * (κmax - κmin)

    main_well(x) = 1.0 - exp(-(x - c_mw)^2 / 2κ^2)
    main_well′(x) = (x - c_mw) * exp(-(x - c_mw)^2 / 2κ^2) / κ^2

    m_V = ((rand(rng) < poneWell) ? 0 : rand(rng, 1:mmax))
    c_V = rand(rng, m_V)
    h_V = (δmin .+ (δmax - δmin) * rand(rng, m_V))
    κ_V = (κmin .+ rand(rng, m_V) * (κmax - κmin)) / m_V

    perturbation(x) = sum((h_V[i] * exp(-(x - c_V[i])^2 / 2κ_V[i]^2) for i in 1:m_V); init = 0.0)
    perturbation′(x) = -sum((h_V[i] * ((x - c_V[i])) * exp(-(x - c_V[i])^2 / 2κ_V[i]^2) / κ_V[i]^2
                             for i in 1:m_V); init = 0.0)

    δE = δmin + (δmax - δmin) * rand(rng)                  # |V(1) - V(0)|

    V0, V1 = main_well(0) + perturbation(0), main_well(1) + perturbation(1)
    b = (V0 < V1) ? (V0 - V1 + δE) : (V0 - V1 - δE)
    X = range(0, 1, n_points)
    a = -minimum(b * X + main_well.(X) + perturbation.(X))
    tmp_max = maximum(@. a + b * X + main_well(X) + perturbation(X))

    V(x) = (a + b * x + main_well(x) + perturbation(x)) / tmp_max
    V′(x) = (b + main_well′(x) + perturbation′(x)) / tmp_max

    logσmin, logσmax = logσlims
    logσscale = logσmin + rand(rng) * (logσmax - logσmin)
    σ = exp(logσscale)

    m_D = ((rand(rng) < pconstD) ? 0 : rand(rng, 1:mmax))
    c_D = rand(rng, m_D)
    h_D = σ * randn(rng, m_D)
    κ_D = (κmin .+ rand(rng, m_D) * (κmax - κmin)) / m_D

    D_perturbation(x) = sum((h_D[i] * exp(-(x - c_D[i])^2 / 2κ_D[i]^2) for i in 1:m_D); init = 0.0)

    d = minimum(D_perturbation.(X))
    D(x) = σ + D_perturbation(x) - d
    D′(x) = -sum((h_D[i] * (x - c_D[i]) * exp(-(x - c_D[i])^2 / 2κ_D[i]^2) / κ_D[i]^2 for i in 1:m_D);
                 init = 0.0)

    return V, D, V′, D′
end
