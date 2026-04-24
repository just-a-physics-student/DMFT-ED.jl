module jED

using Logging, TimerOutputs
using Printf
using Combinatorics
using StaticArrays, OffsetArrays, DataStructures
using LinearAlgebra
using TOML
using LsqFit
# using KrylovKit
using Dispersions
using MultiFloats, GenericLinearAlgebra
using JLD2: jldopen

export Fockstate, Basis, Overlap, Operator, create, ann, create_op, ann_op
export Eigenspace, calc_Hamiltonian
export calc_Z, calc_E, calc_EKin_DMFT, calc_EPot_DMFT, calc_D, calc_Nup, calc_Ndo 
export calc_GF_1, calc_GF_1_inplace, Overlap

# IO
export show_matrix_block, show_energies_states, show_diag_Hamiltonian
export show

export AIM, AIMParams
export Hubbard, Hubbard_Full, Hubbard_Chain
export read_anderson_parameters, write_result
export ScanMode, Fixed_U_increase_T, Fixed_U_decrease_T, Fixed_T_increase_U, Fixed_T_decrease_U

# DMFT
export Σ_from_GImp,
    GWeiss, GWeiss!, GWeiss_from_Δ, GWeiss_from_Imp, Δ_AIM, GLoc, GLoc_MO, fit_AIM_params!
export pick_initial_anderson_parameters, DMFT_Loop

to = TimerOutput()

include("States.jl")
include("Models.jl")
include("Eigenspace.jl")
include("Operators.jl")
include("Observables.jl")
include("GreensFunctions.jl")
include("DMFTLoop.jl")
include("IO.jl")
include("AndersonParamsFit.jl")


end
