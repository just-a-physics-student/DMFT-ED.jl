using Pkg
Pkg.activate(joinpath(@__DIR__,".."))
using jED: AIMParams, pick_initial_anderson_parameters, read_anderson_parameters, DMFT_Loop, write_result, ScanMode, Fixed_U_increase_T, Fixed_U_decrease_T, Fixed_T_increase_U, Fixed_T_decrease_U

using JLD2: jldopen

# example call: rm Desktop/rm_me_fork/* ; julia .julia/dev/DMFT-ED.jl/scripts/dmft-ed.jl 4 1.0 1.0 2.0 1.0 "2Dsc-0.25" 60 4 /home/jan/Desktop/rm_me_fork/
using ArgParse

function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table s begin
        "--start_params_file"
            help = "Path to a file containing Anderson parameters to start the calculation from."
            arg_type = String
            default = ""
        "--overwrite_existing"
            help = "If given, existing result files will be overwritten."
            action = :store_true
        "scan_mode"
            help = "See documentation of `ScanMode`. 1: Fixed U, increase temperature, 2: Fixed U, decrease temperature, 3: Fixed temperature, increase U, 4: Fixed temperature, decrease U"
            arg_type = Int
            required = true
        "constant_param"
            help = "Value of the fixed quantity (hubbard on-site interaction U or inverse temperature) depending on the argument `scan_mode`."
            arg_type = Float64
            required = true
        "lower_bound"
            help = "Lower bound for the variable quantity (hubbard on-site interaction U or inverse temperature) depending on the argument `scan_mode`."
            arg_type = Float64
            required = true
        "upper_bound"
            help = "Upper bound for the variable quantity (hubbard on-site interaction U or inverse temperature) depending on the argument `scan_mode`. Set to the same value as lower bound to enforce a single calculation."
            arg_type = Float64
            required = true
        "step_width"
            help = "Step width to be used between the lower and the upper bound for the variable quantity (hubbard on-site interaction U or inverse temperature) depending on the argument `scan_mode`."
            arg_type = Float64
            required = true
        "lattice_info"
            help = "High-level description of the lattice geometry. See documentation of the k-grid-string from `Dispersions.jl`."
            arg_type = String
            required = true
        "bz_points_per_dim"
            help = "Number of points per dimension to be used for sampling the first Brillouin zone."
            arg_type = Int
            required = true
        "n_bath_sites"
            help = "Number of bath sites in the Anderson Impurity model."
            arg_type = Int
            required = true
        "out_dir"
            help = "Path to an existing directory in which the result should be stored."
            arg_type = String
            required = true
    end
    return parse_args(s)
end

parsed_args = parse_commandline()
println("Parsed args:")
for (arg,val) in parsed_args
    println("  $arg  =>  $val")
end

# dev
n_frequencies::Int  = 2500
max_iterations::Int = 1000000
convergence_paramater::Float64 = 1e-7

# command line
scan_mode           ::ScanMode  = ScanMode(parsed_args["scan_mode"])
constant_param      ::Float64   = parsed_args["constant_param"]
lower_bound         ::Float64   = parsed_args["lower_bound"]
upper_bound         ::Float64   = parsed_args["upper_bound"]
step_width          ::Float64   = parsed_args["step_width"]
lattice_info        ::String    = parsed_args["lattice_info"]
bz_points_per_dim   ::Int       = parsed_args["bz_points_per_dim"]
n_bath_sites        ::Int       = parsed_args["n_bath_sites"]
out_dir             ::String    = parsed_args["out_dir"]
overwrite_existing  ::Bool      = parsed_args["overwrite_existing"]
start_params_file   ::String    = parsed_args["start_params_file"]

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