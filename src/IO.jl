# ==================================================================================================== #
#                                              IO.jl                                                   #
# ---------------------------------------------------------------------------------------------------- #
#   Author          : Julian Stobbe, Steffen Backes                                                    #
# ----------------------------------------- Description ---------------------------------------------- #
#   Input and output operations, including custom printing of types.                                   #
# -------------------------------------------- TODO -------------------------------------------------- #
# ==================================================================================================== #

import Base: show
# ========================= Custom type overloads =========================
function show(io::IO, ::MIME"text/plain", f::Fockstate{Length}) where {Length}
    compact = get(io, :compact, false)
    bb = filter(!isspace, rpad(bitstring(BitVector(f)), Length, "0"))
    N = floor(Int, Length / 2)
    for i = 1:N
        du = parse(Int, bb[i])
        dd = parse(Int, bb[N+i])
        print(io, ("↑"^du) * ("O"^(1 - du)) * ("↓"^dd) * ("O"^(1 - dd)))
        (i < N) && print(io, "-")
    end
end

function show(io::IO, ::MIME"text/plain", b::Basis{Length}) where {Length}
    compact = get(io, :compact, false)
    for (bi, el) in enumerate(b.blocklist)
        block_str = " === Block $(lpad(bi,3))    [N = $(lpad(el[3],2)), S = $(lpad(el[4],3))] ==="
        println(block_str)
        for i in _block_slice(el)
            print(io, "   |          ")
            show(io, MIME"text/plain"(), b.states[i])
            println(io, "")
        end
    end
end
# ======================= Auxilliary Function =======================

function show_matrix_block(H::AbstractMatrix, b::Basis, iBlock::Int)
    start, size, Ni, Si = b.blocklist[iBlock]
    slice = start:start+size-1

    println("(Block for N=$Ni, S=$Si): ")
    show(stdout, "text/plain", H[slice, slice])
    println("\n===============================")
end

"""
    show_diag_Hamiltonian(b::Basis,  es::Eigenspace; io=stdout)

Displays eigenvalues for each block (sorted uin each block).
"""
function show_diag_Hamiltonian(b::Basis,  es::Eigenspace; io=stdout)
    for (bi, el) in enumerate(b.blocklist)
        block_str = " === Block $(lpad(bi,3))    [N = $(lpad(el[3],2)), S = $(lpad(el[4],3))] ==="
        println(io, block_str)
        println(io, "===================================================+")
        display(Diagonal(es.evals[_block_slice(el)] .+ es.E0))
        println(io, "\n====================================================")
        println(io, "")
    end
end

"""
    show_energies_states(b::Basis,  es::Eigenspace; io=stdout, eps_cut=1e-12)

Shows all eigenstates, each block is sorted by eigen energy.
Also displays the eigenvector in terms of the basis vectors for each eigenvalue.

TODO: reasonable formatting solution
"""
function show_energies_states(b::Basis,  es::Eigenspace; io=stdout, eps_cut=1e-12)
    for (bi, el) in enumerate(b.blocklist)
        block_str = " === Block $(lpad(bi,3))    [N = $(lpad(el[3],2)), S = $(lpad(el[4],3))] ==="
        println(block_str)
        bs = _block_slice(el)
        ii = sortperm(es.evals[bs])
        for i in bs[ii]
            print(io, "   | [E=$(lpad(round(es.evals[i] .+ es.E0; digits=4),10))]        ")
            println(io, "")
            print(io, "       |> ")
            start_of_print = true
            for (j,ev_j) in enumerate(es.evecs[i])
                if abs(ev_j) > eps_cut
                    if start_of_print == false
                        print(io, " + ")
                    end
                    start_of_print = false
                    print(io,"$(lpad(round(ev_j; digits=2),5)) x [")
                    show(io, MIME"text/plain"(), b.states[bs[ii][1]+j-1])
                    print(io,"]")
                end
            end
            println(io, "")
        end
    end
end

function write_hubb_andpar(μ::Float64, β::Float64, p::AIMParams, Nν::Int, NBathSites::Int, maxit::Int, abs_conv::Float64, path::String)
    fname = joinpath(path, "hubb.andpar")
    epsk_str = ""
    tpar_str = ""
    for ek in p.ϵₖ
        epsk_str = epsk_str * "$ek\n"
    end
    for vk in p.Vₖ
        tpar_str = tpar_str * "$vk\n"
    end
    out_string = """           ========================================
              HEADER PLACEHOLDER
           ========================================
NSITE     $NBathSites IWMAX $Nν
    $(β)d0, -12.0, 12.0, 0.007
c ns,imaxmu,deltamu, # iterations, conv.param.
   $NBathSites, 0, 0.d0, $maxit, $abs_conv
c ifix(0,1), <n>,   inew, iauto
Eps(k)
$epsk_str tpar(k)
$tpar_str $μ
"""
    open(fname, "w") do f
        write(f, out_string)
    end
end

function write_νFunction(νnGrid::Vector{ComplexF64}, data::Vector{ComplexF64}, fname::String) 
    row_fmt_str = "%27.16f %27.16f %27.16f"
    row_fmt     = jED.Printf.Format(row_fmt_str * "\n")

    open(fname,"w") do f
        for i in 1:length(νnGrid)
            jED.Printf.format(f, row_fmt, imag(νnGrid[i]), real(data[i]), imag(data[i]))
        end
    end
end

@enum ScanMode Fixed_U_increase_T=1 Fixed_U_decrease_T=2 Fixed_T_increase_U=3 Fixed_T_decrease_U=4

function write_result(filepath::String, 
    u::Float64, β::Float64, μ::Float64, p::AIMParams, partition_sum::Float64, G_int::AbstractVector{<:ComplexF64}, Σ_imp::AbstractVector{<:ComplexF64},
    dens::Float64, double_occ::Float64, E_min::Float64, NBathSites::Int, KGridStr::String, converged::Bool, Nk::Int, conv_param::Float64, scan_mode::ScanMode)
    jldopen(filepath, "w") do file
        file["hubbard-u"] = u
        file["inverse-temperature"] = β
        file["chemical-potential"] = μ
        file["bath-energy-levels"] = p.ϵₖ
        file["hybridization-amplitudes"] = p.Vₖ
        file["partition-sum"] = partition_sum
        file["GF-impurity"] = G_int
        file["self-energy-impurity"] = Σ_imp
        file["density"] = dens
        file["double-occupation"] = double_occ
        file["E-shift"] = E_min
        file["n-bath-sites"] = NBathSites
        file["lattice-info"] = KGridStr
        file["bz-points-per-dim"] = Nk
        file["convergence-parameter"] = conv_param
        file["converged"] = converged
        file["scan-mode"] = String(Symbol(scan_mode))
    end
end

"""
    read_anderson_parameters(filepath::String, n_bath_sites::Int)::AIMParams

Reads the anderson parameter from a given file and returns them. Throws an DomainError if the number of bath sites does not match the given number of bath sites.
"""
function read_anderson_parameters(filepath::String, n_bath_sites::Int)::AIMParams
    println("Load start parameters from file: $filepath")
    dmft = jldopen(filepath, "r")
        if dmft["n-bath-sites"] ≠ n_bath_sites
            throw(DomainError(dmft["n-bath-sites"], "does not match the number of bath sites ($n_bath_sites) of the system!"))
        end
        ϵ_bath = dmft["bath-energy-levels"]
        V_hyb  = dmft["hybridization-amplitudes"]
    close(dmft)
    return AIMParams(ϵ_bath, V_hyb)
end