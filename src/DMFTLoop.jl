# ==================================================================================================== #
#                                          DMFTLoop.jl                                                 #
# ---------------------------------------------------------------------------------------------------- #
#   Author          : Julian Stobbe                                                                    #
# ----------------------------------------- Description ---------------------------------------------- #
#  Stub for DMFT Loop related functions                                                                #
# -------------------------------------------- TODO -------------------------------------------------- #
#   This is only a stub, needs to be properly wrapped in types                                         #
# ==================================================================================================== #

const FermionicMatsubaraGrid = OffsetVector{Complex{FPT}} where {FPT<:Real}
const MatsubaraF = OffsetVector{Complex{FPT}} where {FPT<:Real}

"""
    Σ_from_GImp(GWeiss::OffsetVector{ComplexF64, Vector{ComplexF64}}, GImp::OffsetVector{ComplexF64, Vector{ComplexF64}})

Computes self-energy from impurity Green's function (as obtained from a given impurity solver) and the [`Weiss Greens Function`](@ref GWeiss).
"""
function Σ_from_GImp(GWeiss::MatsubaraF, GImp::MatsubaraF)
    return 1 ./ GWeiss .- 1 ./ GImp
end


"""
    GWeiss(Δ::OffsetVector{ComplexF64, Vector{ComplexF64}}, μ::Float64, νnGrid::FermionicMatsubaraGrid)

Computes Weiss Green's frunction from [`hybridization function`](@ref Δ_AIM).
"""
function GWeiss(Δ::MatsubaraF, μ::Float64, νnGrid::FermionicMatsubaraGrid)
    return 1 ./ (νnGrid .+ μ .- Δ)
end

"""
    GWeiss(νnGrid::FermionicMatsubaraGrid, p::AIMParams)
    GWeiss(νnGrid::Vector, μ::Number, ϵₖ::Vector, Vₖ::Vector)
    GWeiss!(target::Vector, νnGrid::Vector, μ::Float64, p::AIMParams)

Computes Weiss Green's frunction from [`Anderson Parameters`](@ref AIMParams).
"""
function GWeiss(νnGrid::FermionicMatsubaraGrid, μ::Float64, p::AIMParams)
    res = similar(νnGrid.parent)
    GWeiss!(res, νnGrid.parent, μ, p.ϵₖ, p.Vₖ)
    return OffsetVector(res, axes(νnGrid))
end

function GWeiss(νnGrid::Vector, μ::Number, ϵₖ::Vector, Vₖ::Vector)
    target = Vector{eltype(νnGrid)}(undef, length(νnGrid))
    for i = 1:length(νnGrid)
        target[i] = (1 / (νnGrid[i] + μ - sum((Vₖ .^ 2) ./ (νnGrid[i] .- ϵₖ))))
    end
    return target
end


function GWeiss_real(νnGrid::Vector, μ::Number, ϵₖ::Vector, Vₖ::Vector)
    target = Vector{eltype(ϵₖ)}(undef, 2*length(νnGrid))
    for i = 1:length(νnGrid)
        target[i] = real(1 / (νnGrid[i] + μ - sum((Vₖ .^ 2) ./ (νnGrid[i] .- ϵₖ))))
    end
    for (i,ii) = enumerate(length(νnGrid)+1:2*length(νnGrid))
        target[ii] = imag(1 / (νnGrid[i] + μ - sum((Vₖ .^ 2) ./ (νnGrid[i] .- ϵₖ))))
    end
    return target
end

function GWeiss!(target::Vector, νnGrid::Vector, μ::Number, ϵₖ::Vector, Vₖ::Vector)
    length(target) != length(νnGrid) &&
        error("νnGrid and target must have the same length!")
    for i = 1:length(νnGrid)
        target[i] = 1 / (νnGrid[i] + μ - sum((Vₖ .^ 2) ./ (νnGrid[i] .- ϵₖ)))
    end
end

"""
    GWeiss_from_Imp(GLoc::MatsubaraF, ΣImp::MatsubaraF)

Compute Updates Weiss Green's function from impurity self-energy via ``[G_\\text{loc} + \\Sigma_\\text{Imp}]^{-1}`` (see [`GLoc`](@ref GLoc) and [`Σ_from_GImp`](@ref Σ_from_GImp)).
"""
function GWeiss_from_Imp(GLoc::MatsubaraF, ΣImp::MatsubaraF)
    return 1 ./ (ΣImp .+ 1 ./ GLoc)
end

"""
    Δ_AIM(νnGrid::FermionicMatsubaraGrid, p::AIMParams)
    Δ_AIM(νnGrid::FermionicMatsubaraGrid, p::Vector{Float64})

Computes hybridization function ``\\sum_p \\frac{V_p^2}{i\\nu_n - \\epsilon_p}`` from [`Anderson Impurity Model Parameters`](@ref AIMParams) with ``p`` bath sites.
"""
function Δ_AIM(νnGrid::FermionicMatsubaraGrid, p::AIMParams)
    return OffsetVector(Δ_AIM(νnGrid.parent, vcat(p.ϵₖ, p.Vₖ)), eachindex(νnGrid))
end

function Δ_AIM(νnGrid::Vector{ComplexF64}, p::Vector{Float64})
    Δ = similar(νnGrid)
    N::Int = floor(Int, length(p) / 2)
    Δint(νn::ComplexF64, p::Vector{Float64})::ComplexF64 =
        conj(sum(p[(N+1):end] .^ 2 ./ (νn .- p[1:N])))
    for νi in eachindex(νnGrid)
        Δ[νi] = Δint(νnGrid[νi], p)
    end
    return Δ
end

function Δ_AIM_real(νnGrid::Vector{ComplexF64}, p::Vector{Float64})
    Δ = similar(νnGrid)
    N::Int = floor(Int, length(p) / 2)
    Δint(νn::ComplexF64, p::Vector{Float64})::ComplexF64 =
        conj(sum(p[(N+1):end] .^ 2 ./ (νn .- p[1:N])))
    for νi in eachindex(νnGrid)
        Δ[νi] = Δint(νnGrid[νi], p)
    end
    return vcat(real.(Δ), imag.(Δ))
end


"""
    Δ_from_GWeiss(GWeiss::MatsubaraF, μ::Float64, νnGrid::FermionicMatsubaraGrid)

Computes hybridization function from Weiss Green's function via ``\\Delta^{\\nu} = i\\nu_n + \\mu - \\left(\\mathcal{G}^{\\nu}_0\\right^{-1}``.
"""
function Δ_from_GWeiss(GWeiss::MatsubaraF, μ::Float64, νnGrid::FermionicMatsubaraGrid)
    return νnGrid .- μ .- 1 ./ GWeiss
end

function GLoc_MO_old2(
    ΣImp::MatsubaraF,
    μ::Float64,
    νnGrid::FermionicMatsubaraGrid,
    kG::KGrid,
)
    @assert length(νnGrid) <= length(ΣImp)
    GLoc = zero(ΣImp)
    tmp = dispersion(kG)

    iOrb::Int = 1
    for (ki, kMult) in enumerate(kG.kMult)
        for νi in eachindex(νnGrid)
            νn = νnGrid[νi]
            @inbounds GLoc[νi] += kMult * (((μ.+νn-ΣImp[νi])*I+tmp[:, :, ki])\I)[iOrb, iOrb]
        end
    end
    GLoc = GLoc ./ Nk(kG)
    GLoc = 1 ./ (1 ./ GLoc .+ ΣImp)
    return GLoc
end


"""
    GLoc(ΣImp::MatsubaraF, μ::Float64, νnGrid::FermionicMatsubaraGrid, kG::KGrid)

Compute local Green's function ``\\int dk [i\\nu_n + \\mu + \\epsilon_k - \\Sigma_\\text{Imp}(i\\nu_n)]^{-1}``.
TODO: simplify -conj!!!
"""
function GLoc(ΣImp::MatsubaraF, μ::Float64, νnGrid::FermionicMatsubaraGrid, kG::KGrid)
    GLoc = similar(ΣImp)
    tmp = μ .- dispersion(kG)

    # TODO: this is only here for testing purposes! Remove and implement multi-orbital case
    if typeof(kG).parameters[1] <: Hofstadter
        error("Call GLoc_MO for Hofstaedter model!")
    end
    for νi in eachindex(νnGrid)
        νn = νnGrid[νi]
        GLoc[νi] = kintegrate(kG, 1 ./ (tmp .+ νn .- ΣImp[νi]))
    end
    GLoc = 1 ./ (1 ./ GLoc .+ ΣImp)
    return GLoc
end

"""
    DMFT_Loop(U::Float64, μ::Float64, β::Float64, NBathSites::Int, KGridStr::String; 
              Nk::Int=60, Nν::Int=1000, α::Float64=0.7, abs_conv::Float64=1e-8, ϵ_cut::Float64=1e-12, maxit = 20)

Arguments:
----------
    - U::Float64              : Hubbard U
    - μ::Float64              : chemical potential
    - β::Float64              : inverse temperature
    - p::AIMParams            : initial anderson parameters
    - NBathSites::Int         : number of bath sites
    - KGridStr::String        : K-Grid String (see Dispersions.jl)
    - Nk::Int=60              : Number of K-Points for GLoc
    - Nν::Int=1000            : Number of Matsubara frequencies
    - α::Float64=0.7          : Mixing (1 -> no mixing)
    - abs_conv::Float64=1e-8  : Absolute difference of Anderson parameters that if fallen short of leads to termination of the algorithm
    - ϵ_cut::Float64=1e-12    : Terms smaller than this are not considered for the impurity Green's function
    - maxit::Int=20           : Maximum number of DMFT iterations

Returns:
----------
    - p       : Anderson parameters
    - νnGrid  : Matsubara grid
    - GImp    : Impurity Green's function
    - ΣImp    : Impurity self-energy
"""
function DMFT_Loop(U::Float64, μ::Float64, β::Float64, p::AIMParams, KGridStr::String; 
                   Nk::Int=60, Nν::Int=1000, α::Float64=0.7, abs_conv::Float64=1e-8, ϵ_cut::Float64=1e-12, maxit::Int=20, checkpointfile::String="")
    println(" ======== U = $U / μ = $μ / β = $β / NB = $(length(p.ϵₖ)) / INIT ======== ")
    println("Solution using Lsq:    ϵₖ = $(lpad.(round.(p.ϵₖ,digits=4),9)...)")
    println("                       Vₖ = $(lpad.(round.(p.Vₖ,digits=4),9)...)")
    GImp_i = nothing
    GImp_i_old = nothing
    ΣImp_i = nothing
    dens   = Inf
    Z      = Inf
    done   = false
    i      = 1

    kG     = jED.gen_kGrid(KGridStr, Nk)
    basis  = jED.Basis(length(p.Vₖ) + 1);
    overlap= Overlap(basis, create_op(basis, 1)) # optional
    νnGrid = jED.OffsetVector([1im * (2*n+1)*π/β for n in 0:Nν-1], 0:Nν-1)

    converged::Bool = false
    E_smallest::Float64 = Inf64
    D::Float64 = Inf64

    checkpoint = !isempty(checkpointfile)
    
    while !done
        model  = AIM(p.ϵₖ, p.Vₖ, μ, U)
        G0W    = GWeiss(νnGrid, μ, p)
        es     = Eigenspace(model, basis);
        isnothing(GImp_i_old) ? GImp_i_old = deepcopy(GImp_i) : copyto!(GImp_i_old, GImp_i)
        GImp_i, dens = calc_GF_1(basis, es, νnGrid, β, ϵ_cut=ϵ_cut, overlap=overlap)
        !isnothing(GImp_i_old) && (GImp_i = α .* GImp_i .+ (1-α) .* GImp_i_old)
        ΣImp_i = Σ_from_GImp(G0W, GImp_i)
        Z = calc_Z(es, β)

        GLoc_i = GLoc(ΣImp_i, μ, νnGrid, kG)
        p_old = deepcopy(p)
        fit_AIM_params!(p, GLoc_i, μ, νnGrid)
        
        converged = (sum(abs.(p_old.ϵₖ .- p.ϵₖ)) + sum(abs.(p_old.Vₖ .- p.Vₖ))) < abs_conv
        println(" iteration $i: $(sum(abs.(p_old.ϵₖ .- p.ϵₖ)) + sum(abs.(p_old.Vₖ .- p.Vₖ)))")
        println("   ϵₖ = $(lpad.(round.(p.ϵₖ,digits=4),9)...)")
        println("   Vₖ = $(lpad.(round.(p.Vₖ,digits=4),9)...)")
        println("   --> sum(Vₖ²) = $(sum(p.Vₖ .^ 2)) // Z = $Z")

        if converged || i >= maxit
            GImp_i, dens = calc_GF_1(basis, es, νnGrid, β, ϵ_cut=ϵ_cut, overlap=overlap, with_density=true)
            Z = calc_Z(es, β)
            E_smallest = es.E0
            D = jED.calc_D(es, β, basis, model.impuritySiteIndex)
            done = true
        end
        if checkpoint
            println("write checkpoint file")
            write_anderson_parameters(checkpointfile, p, length(p.Vₖ), converged)
        end
        i += 1
    end
    return p, GImp_i, ΣImp_i, Z, E_smallest, D, dens, converged, νnGrid
end


"""
    pick_initial_anderson_parameters(U::Float64, NBathSites::Int)::AIMParams

Calculate a anderson parameters to start a DMFT calculation with. This might serve as a starting point when a new system is investigated.
"""
function pick_initial_anderson_parameters(U::Float64, NBathSites::Int)::AIMParams
    ϵₖ = [iseven(NBathSites) || i != ceil(Int, NBathSites/2) ? (U/2)/(i-NBathSites/2-1/2) : 0 for i in 1:NBathSites]
    Vₖ = [1/(4*NBathSites) for i in 1:NBathSites]
    return AIMParams(ϵₖ, Vₖ)
end

function DMFT_Loop(U::Float64, μ::Float64, β::Float64, NBathSites::Int, KGridStr::String;
    Nk::Int=60, Nν::Int=1000, α::Float64=0.7, abs_conv::Float64=1e-8, ϵ_cut::Float64=1e-12, maxit::Int=20)
    p::AIMParams = pick_initial_anderson_parameters(U, NBathSites)
    return DMFT_Loop(U, μ, β, p, KGridStr; Nk=Nk, Nν=Nν, α=α, abs_conv=abs_conv, ϵ_cut=ϵ_cut, maxit=maxit)
end