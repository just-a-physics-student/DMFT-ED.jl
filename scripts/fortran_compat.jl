using Pkg
Pkg.activate(joinpath(@__DIR__,".."))
using jED

# example call: rm /home/jan/Desktop/rm_me/* ; julia /home/jan/.julia/dev/DMFT-ED.jl/scripts/fortran_compat.jl 1.0 1.0 0.5 4 2Dsc-0.25 /home/jan/Desktop/rm_me

length(ARGS) < 6 && error("Please proivide U/beta/mu/NBathSites/KGridStr/Path as arguments to the script!")
U = parse(Float64, ARGS[1])
β = parse(Float64, ARGS[2])
μ = parse(Float64, ARGS[3])
NBathSites = parse(Int, ARGS[4])
KGridStr   = ARGS[5]
path       = ARGS[6]

Nν::Int    = 3000 
maxit::Int = 10
abs_conv::Float64 = 1e-9

"""
    DMFT_Loop(U::Float64, μ::Float64, β::Float64, NBathSites::Int, KGridStr::String; 
              Nk::Int=60, Nν::Int=1000, α::Float64=0.7, abs_conv::Float64=1e-8, ϵ_cut::Float64=1e-12, maxit = 20)

Arguments:
----------
    - U::Float64              : Hubbard U
    - μ::Float64              : chemical potential
    - β::Float64              : inverse temperature
    - NBathSites::Int         : number of bath sites
    - KGridStr::String        : K-Grid String (see Dispersions.jl)
    - Nk::Int=60              : Number of K-Points for GLoc
    - Nν::Int=1000            : Number of Matsubara frequencies
    - α::Float64=0.7          : Mixing (1 -> no mixing)
    - abs_conv::Float64=1e-8  : Difference of Anderson parameters to last iteration for convergence
    - ϵ_cut::Float64=1e-12    : Terms smaller than this are not considered for the impurity Green's function
    - maxit::Int=20           : Maximum number of DMFT iterations

Returns:
----------
    - p       : Anderson parameters
    - νnGrid  : Matsubara grid
    - GImp    : Impurity Green's function
    - ΣImp    : Impurity self-energy
"""
function DMFT_Loop(U::Float64, μ::Float64, β::Float64, NBathSites::Int, KGridStr::String; 
                   Nk::Int=60, Nν::Int=1000, α::Float64=0.7, abs_conv::Float64=1e-8, ϵ_cut::Float64=1e-12, maxit::Int=20)
    ϵₖ = [iseven(NBathSites) || i != ceil(Int, NBathSites/2) ? (U/2)/(i-NBathSites/2-1/2) : 0 for i in 1:NBathSites]
    Vₖ = [1/(4*NBathSites) for i in 1:NBathSites]
    p  = AIMParams(ϵₖ, Vₖ)
    println(" ======== U = $U / μ = $μ / β = $β / NB = $(length(ϵₖ)) / INIT ======== ")
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
    basis  = jED.Basis(length(Vₖ) + 1);
    overlap= Overlap(basis, create_op(basis, 1)) # optional
    νnGrid = jED.OffsetVector([1im * (2*n+1)*π/β for n in 0:Nν-1], 0:Nν-1)
        
    while !done
        model  = AIM(p.ϵₖ, p.Vₖ, μ, U)
        G0W    = GWeiss(νnGrid, μ, p)
        es     = Eigenspace(model, basis);
        isnothing(GImp_i_old) ? GImp_i_old = deepcopy(GImp_i) : copyto!(GImp_i_old, GImp_i)
        println("     Calculating GImp")
        GImp_i, dens = calc_GF_1(basis, es, νnGrid, β, ϵ_cut=ϵ_cut, overlap=overlap)
        !isnothing(GImp_i_old) && (GImp_i = α .* GImp_i .+ (1-α) .* GImp_i_old)
        ΣImp_i = Σ_from_GImp(G0W, GImp_i)
        Z = calc_Z(es, β)

        GLoc_i = GLoc(ΣImp_i, μ, νnGrid, kG)
        p_old = deepcopy(p)
        fit_AIM_params!(p, GLoc_i, μ, νnGrid)
        println(" ======== U = $U / μ = $μ / β = $β / NB = $(length(ϵₖ)) / it = $i ======== ")
        println("Solution using Lsq:    ϵₖ = $(lpad.(round.(p.ϵₖ,digits=4),9)...)")
        println("                       Vₖ = $(lpad.(round.(p.Vₖ,digits=4),9)...)")
        println(" -> sum(Vₖ²) = $(sum(p.Vₖ .^ 2)) // Z = $Z")
        if ((sum(abs.(p_old.ϵₖ .- p.ϵₖ)) + sum(abs.(p_old.Vₖ .- p.Vₖ))) < abs_conv) || i >= maxit
            GImp_i, dens = calc_GF_1(basis, es, νnGrid, β, ϵ_cut=ϵ_cut, overlap=overlap, with_density=true)
            Z = calc_Z(es, β)
            done = true
        end
        i += 1
    end

    return p, νnGrid, GImp_i, ΣImp_i, Z, dens
end


# ==================== IO Functions ====================
function write_hubb_andpar(p::AIMParams)
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

# ==================== Calculation and Output ====================
p, νnGrid, GImp, ΣImp, Z, dens = DMFT_Loop(U, μ, β, NBathSites, KGridStr, abs_conv=abs_conv, Nν=Nν,  maxit = maxit)
G0W    = GWeiss(νnGrid, μ, p)

write_hubb_andpar(p)
write_νFunction(νnGrid.parent, GImp.parent, joinpath(path, "gm_wim"))  
write_νFunction(νnGrid.parent, 1 ./ G0W.parent, joinpath(path, "g0m"))  
open(joinpath(path,"zpart.dat"), "w") do f
    write(f, "$Z")
end
open(joinpath(path,"densimp.dat"), "w") do f
    write(f, "$dens")
end

# TODO: write zpart.dat, densimp.dat (??)
