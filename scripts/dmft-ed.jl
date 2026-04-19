using Pkg
Pkg.activate(joinpath(@__DIR__,".."))
using jED

using JLD2: jldopen

# example call: rm Desktop/rm_me_fork/* ; julia .julia/dev/DMFT-ED.jl/scripts/dmft-ed.jl 4 1.0 1.0 2.0 1.0 2Dsc-0.25 60 4 /home/jan/Desktop/rm_me_fork/

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
                   Nk::Int=60, Nν::Int=1000, α::Float64=0.7, abs_conv::Float64=1e-8, ϵ_cut::Float64=1e-12, maxit::Int=20)
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


# ==================== IO Functions ====================
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


# dev
n_frequencies::Int  = 2500
max_iterations::Int = 1000000
convergence_paramater::Float64 = 1e-7

# command line
length(ARGS) < 9 && error("Not enough arguments given!")
scan_mode           = ScanMode(parse(Int, ARGS[1]))
constant_param      = parse(Float64, ARGS[2]) # hubbard on-site interaction or temperature: depending on the scan-mode
lower_bound         = parse(Float64, ARGS[3]) # hubbard on-site interaction or temperature: depending on the scan-mode
upper_bound         = parse(Float64, ARGS[4]) # hubbard on-site interaction or temperature: depending on the scan-mode
step_width          = parse(Float64, ARGS[5])
lattice_info        = ARGS[6]
bz_points_per_dim   = parse(Int, ARGS[7])
n_bath_sites        = parse(Int, ARGS[8])
out_dir             = ARGS[9]
overwrite_existing  = (length(ARGS) >= 10) ? parse(Bool, ARGS[10]) : false
start_params_file   = (length(ARGS) >= 11) ? ARGS[11] : ""

# define path in (U,T) plane
if lower_bound == upper_bound # single calculation
    scan_values = [lower_bound]
else # multiple calculations
    scan_values = LinRange(lower_bound, upper_bound, Int(round((upper_bound - lower_bound) / step_width) + 1)) # ascending order
end
hubbard_u_values = []
inverse_temperature_values = []
if scan_mode in [Fixed_U_increase_T, Fixed_U_decrease_T]
    push!(hubbard_u_values, constant_param)
    append!(inverse_temperature_values, scan_values)
    if scan_mode == Fixed_U_increase_T
        reverse!(inverse_temperature_values)
    end
elseif scan_mode in [Fixed_T_increase_U, Fixed_T_decrease_U]
    push!(inverse_temperature_values, constant_param)
    append!(hubbard_u_values, scan_values)
    if scan_mode == Fixed_T_decrease_U
        reverse!(hubbard_u_values)
    end
end

# select initial anderson parameters
if isempty(start_params_file)
    anderson_parameters::AIMParams = pick_initial_anderson_parameters(first(hubbard_u_values), n_bath_sites)
else
    anderson_parameters::AIMParams = read_anderson_parameters(start_params_file, n_bath_sites)
end

# function call via double loop so that the call statement appears exactly once
n_calc::Int = 1
for hubbard_u in hubbard_u_values
    chemical_potential::Float64  = hubbard_u / 2 # half filling
    for inverse_temperature in inverse_temperature_values
        global n_calc, anderson_parameters
        # prepare result file
        fname = "dmft_calc_$(n_calc)_scanmode$(Int(scan_mode))_bath$(n_bath_sites)_u$(hubbard_u)_beta$(inverse_temperature)_bz$(bz_points_per_dim).jld2" # enumerate calculations to simplify restart feature
        out_file_path = joinpath(out_dir, fname)
        if isfile(out_file_path)
            println("Result file already exists: $(out_file_path)")
            if !overwrite_existing
                throw(ArgumentError("Stop here! In order to allow overwriting the existing result file, provide cmd argument 10!"))
            end
            println("Overwriting the result file was explicitly allowed! Resume calculation from last result.")
            anderson_parameters = read_anderson_parameters(out_file_path, n_bath_sites)
        end
        
        # run calculation
        println("Start DMFT calc $(n_calc): U = $(hubbard_u) , β = $(inverse_temperature)")
        @time anderson_parameters, GF_imp, Σ_imp, partition_sum, E_min, double_occupancy, density, converged, νnGrid = DMFT_Loop(
            hubbard_u, chemical_potential, inverse_temperature,
            anderson_parameters, lattice_info, Nν=n_frequencies,
            abs_conv=convergence_paramater, maxit=max_iterations)
        
        # write result
        write_result(out_file_path, hubbard_u, inverse_temperature, chemical_potential, anderson_parameters, partition_sum, GF_imp.parent, Σ_imp.parent,
            density, double_occupancy, E_min, n_bath_sites, lattice_info, converged, bz_points_per_dim, convergence_paramater, scan_mode)

        n_calc += 1
    end
end