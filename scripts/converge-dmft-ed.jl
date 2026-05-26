using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using JLD2: jldopen
using jED

# example call: julia +1.11 /home/jan/.julia/dev/DMFT-ED.jl/scripts/converge-dmft-ed.jl "1e-10" /home/jan/Desktop/rm_me_fork/dmft_calc_1_scanmode4_bath4_u3.0_beta1.0_bz20.jld2 /home/jan/Desktop/rm_me_fork/converged.jld2 
using ArgParse

function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table s begin
        "convergence_parameter"
            help = "The value of the convergence parameter that should be exceeded."
            arg_type = Float64
            required = true
        "in_file"
            help = "Path to an existing DMFT result"
            arg_type = String
            required = true
        "out_file"
            help = "Path to save the output file to."
            arg_type = String
            required = true
        "--max_iterations"
            help = "Upper bound for the number of DMFT iterations to be executed. A fully reslut will be written once this number is exceeded. Defaults t0 250."
            arg_type = Int
            default = 100000
        "--bz_points_per_dim"
            help = "Number of points per dimension to be used for sampling the first Brillouin zone. Use-case: start calculation with the same Anderson Parameters but sample the frist Brillouin Zone more/less dense."
            arg_type = Int
            default = 0
        "--n_frequencies"
            help = "Number of positive Matsubara frequencies to be used. Use-case: start calculation with the same Anderson Parameters but increase the number of frequencies considered."
            arg_type = Int
            default = 0
        "--resume"
            help = "Instructs the script to continue from where a previous run was interrupted. Converged output files will remain unchanged. The program will refuse to overwrite an existing output file unless this flag is given."
            action = :store_true
    end
    return parse_args(s)
end

parsed_args = parse_commandline()
println("Parsed args:")
for (arg, val) in parsed_args
    println("  $arg  =>  $val")
end

# command line
convergence_parameter::Float64 = parsed_args["convergence_parameter"]
in_file              ::String  = parsed_args["in_file"]
out_file             ::String  = parsed_args["out_file"]
max_iterations       ::Int     = parsed_args["max_iterations"]
bz_points_per_dim    ::Int     = parsed_args["bz_points_per_dim"]
n_frequencies        ::Int     = parsed_args["n_frequencies"]
resume               ::Bool    = parsed_args["resume"]

!resume && isfile(out_file) && throw(ArgumentError("Out file already exists! If you try to resume this script from a previous run, use the --resume flag!"))
if (read_convergence_parameter(in_file) < convergence_parameter) && (
    (bz_points_per_dim == 0) || (n_frequencies == 0)) # true if sampling of the brillouin zone or the number of frequencies is altered
    println("Input calculation already converged. Perform an iteration and write result.")
end

n_frequencies = (n_frequencies == 0) ? read_n_frequencies(in_file) : n_frequencies
bz_points_per_dim = (bz_points_per_dim == 0) ? read_bz_sampling(in_file) : bz_points_per_dim

lattice_info, hubbard_u, inverse_temperature, chemical_potential, anderson_parameters = read_preliminary_result(in_file)

anderson_parameters, GF_imp, Σ_imp, partition_sum, E_min, double_occupancy, density, converged, νnGrid, convergence_parameter_out = DMFT_Loop(
    hubbard_u, chemical_potential, inverse_temperature,
    anderson_parameters, lattice_info, Nk=bz_points_per_dim, Nν=n_frequencies,
    abs_conv=convergence_parameter, maxit=max_iterations, checkpointfile=out_file)

# write result
n_bath_sites::Int = length(anderson_parameters.ϵₖ)
write_result(out_file, hubbard_u, inverse_temperature, chemical_potential, anderson_parameters, partition_sum, GF_imp.parent, Σ_imp.parent,
    density, double_occupancy, E_min, n_bath_sites, lattice_info, converged, bz_points_per_dim, convergence_parameter_out)