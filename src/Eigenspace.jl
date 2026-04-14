# ==================================================================================================== #
#                                          Eigenspace.jl                                               #
# ---------------------------------------------------------------------------------------------------- #
#   Author          : Julian Stobbe                                                                    #
# ----------------------------------------- Description ---------------------------------------------- #
#   Types and constructors for an Eigenspace, given a model Hamiltonian and states.                    #
# -------------------------------------------- TODO -------------------------------------------------- #
#   _H_nn is not complete (J term missing)                                                             #
# ==================================================================================================== #

const Hamiltonian = Hermitian{ComplexF64,Matrix{ComplexF64}}
const HamiltonianReal = Symmetric{Float64,Matrix{Float64}}


# ============================================= Eigenspace ===========================================
"""
    Eigenspace

Containes Eigenvalues and Eigenvectors grouped into blocks with are each ordered by magnitude of Eigenvalues.
The blocks are part of the [`Basis`](@ref Basis).

Fields
-------------
- **`evals`**     : Eigenvalues
- **`evecs`**     : Eigenvectors
- **`E0`**        : smallest Eigenvalue
"""
struct Eigenspace{FPT<:Real}
    evals::Vector{FPT}
    evecs::Vector{Vector{FPT}}
    E0::FPT
end

"""
    Eigenspace(model::Model, basis::Basis; verbose::Bool = true, FPT::Type{FPTi} = eltype(model.tMatrix))

Constructs the Eigenspace for [`Model`](@ref Model) over given [`Basis`](@ref Basis) by diagonalizing the Hamiltonian (see also [`calc_Hamiltonian`](@ref calc_Hamiltonian)) for each block.
"""
function Eigenspace(
    model::Model,
    basis::Basis;
    verbose::Bool=true,
    FPT::Type{FPTi}=eltype(model.tMatrix),
) where {FPTi<:Real}

    EVecType = typeof(model).parameters[2]
    evals = Vector{FPT}(undef, length(basis.states))
    evecs = Vector{Vector{EVecType}}(undef, length(basis.states))

    # verbose && print("Generating Eigenspace:   0.0% done.")
    for el in basis.blocklist
        slice = _block_slice(el)
        Hi = calc_Hamiltonian(model, basis.states[slice]; FPT=FPT)
        tmp = eigen(Hi)
        evals[slice] .= tmp.values
        for i = 1:length(tmp.values)
            evecs[first(slice)+i-1] = tmp.vectors[:, i]
        end
        verbose && (
            done = lpad(
                round(100 * (el[1] + el[2]) / length(basis.states), digits=1),5," ",
            )
        )
        # verbose && print("\rGenerating Eigenspace: $(done)% done.")
    end
    # verbose && println("\rEigenspace generated!                  ")
    E0 = minimum(evals)

    return Eigenspace{FPT}(evals .- E0, evecs, E0)
end

function Eigenspace_L(model::Model, basis::Basis)

    EVecType = typeof(model).parameters[2]
    evals = Inf .* ones(length(basis.states))
    evecs = Vector{Vector{EVecType}}(undef, length(basis.states))
    issymmetric = eltype(model.tMatrix) === Float64 ? true : false

    # print("Generating Eigenspace:   0.0% done.")
    for el in basis.blocklist
        slice = _block_slice(el)
        Hi = calc_Hamiltonian(model, basis.states[slice])
        krylov_dim = size(Hi, 1) > 200 ? floor(Int, size(Hi, 1) / 10) : size(Hi, 1)
        # values, vectors, conv = eigsolve(Hi, ishermitian=true)
        values, vectors, conv = eigsolve(
            Hi,
            rand(Float64, size(Hi, 1)),
            krylov_dim,
            :SR,
            krylovdim=krylov_dim,
            ishermitian=true,
            issymmetric=issymmetric,
        )
        nv = conv.converged
        evals[first(slice):first(slice)+nv-1] .= values[1:nv]
        for i = 1:nv
            evecs[first(slice)+i-1] = vectors[i]
        end
        done = lpad(round(100 * (el[1] + el[2]) / length(basis.states), digits=1), 5, " ")
        # print("\rGenerating Eigenspace: $(done)% done.")
    end
    # println("\rEigenspace generated!                  ")
    E0 = minimum(evals)

    return evals, evecs, E0
end

# ============================================ Hamiltonian ===========================================
"""
    calc_Hamiltonian(model::Model, basis::Basis)

Calculates the Hamiltonian for a given 
  - `model`, see for example [`AIM`](@ref AIM))) in a 
  - `basis`, see [`Basis`](@ref Basis)  
"""
function calc_Hamiltonian(
    model::Model,
    states::Vector{Fockstate{NSites}};
    FPT::Type{FPTi}=eltype(model.tMatrix),
) where {NSites,FPTi<:Real}
    Hsize = length(states)
    H_int = Matrix{FPT}(undef, Hsize, Hsize)
    for i = 1:Hsize
        H_int[i, i] =
            _H_nn(states[i], states[i], model.UMatrix) +
            _H_CDagC(states[i], states[i], model.tMatrix)
        # We are generating a Hermitian/Symmetric matrix and only need to store the upper triangular part
        for j = i+1:Hsize
            val = _H_CDagC(states[i], states[j], model.tMatrix)
            H_int[i, j] = val
        end
    end
    return FPT isa Real ? Symmetric(H_int, :U) : Hermitian(H_int, :U)
end

calc_Hamiltonian(model::Model, basis::Basis) = calc_Hamiltonian(model, basis.states)



# ======================================== Auxilliary Functions ======================================
"""
    _H_CDag_C(istate, jstate, tMatrix)

Returns the hopping contribution for states ``\\sum_{i,j} \\langle i | T | j \\rangle``, with T being the hopping matrix `tmatrix` and
the states i and j given by `istate` and `jstate`.
"""
function _H_CDagC(bra::Fockstate, ket::Fockstate, tMatrix::SMatrix)
    T = eltype(tMatrix)
    res::T = zero(T)
    NFlavors = 2
    NSites = size(tMatrix, 1)

    for i = 1:NSites
        for j = 1:NSites
            tval = tMatrix[i, j]
            if tval != 0
                for f = 1:NFlavors
                    annInd = NSites * (f - 1) + i
                    createInd = NSites * (f - 1) + j
                    res += tval * overlap_cdagger_c(bra, createInd, ket, annInd)
                end
            end
        end
    end
    return res
end

"""
    _H_nn(istate, jstate, tMatrix)

Returns the density-density contribution for states ``\\sum_{i} \\langle i | U | i \\rangle``, with T being the hopping matrix `tmatrix` and
the states i and j given by `istate` and `jstate`.
"""
function _H_nn(bra::Fockstate, ket::Fockstate, UMatrix::SMatrix)
    T = eltype(UMatrix)
    res::T = zero(T)
    NFlavors = 2
    NSites = size(UMatrix, 1)

    for i = 1:NSites
        uval = UMatrix[i, i]
        i1 = i
        i2 = NSites + i
        n1 = overlap_ni_nj(bra, ket, i1, i2)
        res += uval * n1
    end
    return res
end
