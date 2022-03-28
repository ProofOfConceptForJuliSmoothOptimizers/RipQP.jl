module RipQP

using DelimitedFiles, LinearAlgebra, MatrixMarket, Quadmath, SparseArrays, Statistics

using SolverParameters

using Krylov,
  LDLFactorizations,
  LinearOperators,
  LLSModels,
  NLPModelsModifiers,
  QuadraticModels,
  SolverCore,
  SparseMatricesCOO

using JSOSolvers:AbstractOptSolver

using Requires
function __init__()
  @require CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba" include("gpu_utils.jl")
end

export ripqp, RipQPSolver

include("types_definition.jl")
include("iterations/iterations.jl")
include("refinement.jl")
include("data_initialization.jl")
include("starting_points.jl")
include("scaling.jl")
include("multi_precision.jl")
include("utils.jl")

mutable struct RipQPSolver{T0 <: Real, Int} <: AbstractOptSolver{T0, Int}
  iconf::InputConfig{Int}
  itol::InputTol{T0, Int}
  parameters::Dict{String, AlgorithmicParameter}
  function RipQPSolver(iconf::InputConfig{Int}, itol::InputTol{T0, Int}, parameters::Dict{String, AlgorithmicParameter}) where {T0 <: Real}
    new{T0, Int}(iconf, itol, parameters)
  end
end

function RipQPSolver(::QuadraticModel{T0};iconf=InputConfig(), itol=InputTol(T0), parameters=get_default_parameters()) where {T0 <: Real}
  return RipQPSolver(iconf, itol, parameters)
end

function get_default_parameters()
  scaling_param = AlgorithmicParameter(true, BinaryRange(), "scaling")
  kc_param = AlgorithmicParameter(-1, IntegerRange(-1, 10), "kc")
  presolve_param = AlgorithmicParameter(true, BinaryRange(), "presolve")
  atol_min_param = AlgorithmicParameter(1.0e-10, RealInterval(1.0e-10, 1.0e-7), "atol_min")
  ρ0_param = AlgorithmicParameter(sqrt(eps()) * 1e5, RealInterval(0.0, 1.0), "ρ0")
  δ0_param = AlgorithmicParameter(sqrt(eps()) * 1e5, RealInterval(0.0, 1.0), "δ0")

  return Dict{String, AlgorithmicParameter}(
    "scaling" => scaling_param,
    "kc" => kc_param,
    "presolve" => presolve_param,
    "atol_min" => atol_min_param,
    "ρ0" => ρ0_param,
    "δ0" => δ0_param,
    "memory" => AlgorithmicParameter(20, IntegerRange(15, 25), "memory")
  )
end

"""
    stats = ripqp(QM :: QuadraticModel{T0}; iconf :: InputConfig{Int} = InputConfig(),
                  itol :: InputTol{T0, Int} = InputTol(T0),
                  display :: Bool = true) where {T0<:Real}

Minimize a convex quadratic problem. Algorithm stops when the criteria in pdd, rb, and rc are valid.
Returns a [GenericExecutionStats](https://juliasmoothoptimizers.github.io/SolverCore.jl/dev/reference/#SolverCore.GenericExecutionStats) 
containing information about the solved problem.

- `QM :: QuadraticModel`: problem to solve
- `iconf :: InputConfig{Int}`: input RipQP configuration. See [`RipQP.InputConfig`](@ref).
- `itol :: InputTol{T, Int}` input Tolerances for the stopping criteria. See [`RipQP.InputTol`](@ref).
- `display::Bool`: activate/deactivate iteration data display

You can also use `ripqp` to solve a [LLSModel](https://juliasmoothoptimizers.github.io/LLSModels.jl/stable/#LLSModels.LLSModel):

    stats = ripqp(LLS :: LLSModel{T0}; iconf :: InputConfig{Int} = InputConfig(),
                  itol :: InputTol{T0, Int} = InputTol(T0),
                  display :: Bool = true) where {T0<:Real}
"""
function ripqp(QM0::QuadraticModel{T0}, params::Vector{AlgorithmicParameter};kwargs...) where {T0 <: Real}
  solver = RipQPSolver(QM0)
  solver.itol = InputTol(T0;max_time=120.0)
  solver.parameters = Dict{String, AlgorithmicParameter}(name(p) => p for p in params)
  ripqp(QM0, solver;kwargs...)
end

function ripqp(QM0::QuadraticModel{T0}, solver::RipQPSolver{T0, Int};kwargs...) where {T0 <: Real}
  params = solver.parameters
  solver_params = K2KrylovParams(:L, :minres, :Identity,
    true, false,
    1.0e-4, 1.0e-4, 
    default(params["atol_min"]), 1.0e-10,
    default(params["ρ0"]), default(params["δ0"]),
    1e2 * sqrt(eps()), 1e2 * sqrt(eps()),
    default(params["memory"])
    )
  solver.iconf = InputConfig(;
  scaling=default(params["scaling"]),
  kc=default(params["kc"]),
  presolve=default(params["presolve"]),
    sp=solver_params
  )
  return ripqp(QM0;iconf=solver.iconf, itol=solver.itol, kwargs...)
end

function ripqp(
  QM0::QuadraticModel{T0};
  iconf::InputConfig{Int} = InputConfig(),
  itol::InputTol{T0, Int} = InputTol(T0),
  display::Bool = true,
) where {T0 <: Real}
  start_time = time()
  elapsed_time = 0.0
  # conversion function if QM.data.H and QM.data.A are not in the type required by iconf.sp
  QM0 = convert_QM(QM0, iconf, display)

  if iconf.presolve
    QM = presolve(QM0)
  else
    QM = QM0
  end

  # allocate workspace
  sc, idi, fd_T0, id, ϵ, res, itd, dda, pt, sd, spd, cnts, T =
    allocate_workspace(QM, iconf, itol, start_time, T0)

  if iconf.scaling
    scaling_Ruiz!(fd_T0, id, sd, T0(1.0e-3))
  end

  # extra workspace for multi mode
  if iconf.mode == :multi
    fd32, ϵ32, T = allocate_extra_workspace_32(itol, iconf, fd_T0)
    if T0 == Float128
      fd64, ϵ64, T = allocate_extra_workspace_64(itol, iconf, fd_T0)
    end
  end

  # initialize
  if iconf.mode == :multi
    pad = initialize!(fd32, id, res, itd, dda, pt, spd, ϵ32, sc, iconf, cnts, T0)
    set_tol_residuals!(ϵ, T0(res.rbNorm), T0(res.rcNorm))
    if T0 == Float128
      set_tol_residuals!(ϵ64, Float64(res.rbNorm), Float64(res.rcNorm))
      T = Float32
    end
  elseif iconf.mode == :mono
    pad = initialize!(fd_T0, id, res, itd, dda, pt, spd, ϵ, sc, iconf, cnts, T0)
  end

  Δt = time() - start_time
  sc.tired = Δt > itol.max_time

  # display
  if display == true
    @info log_header(
      [:k, :pri_obj, :pdd, :rbNorm, :rcNorm, :α_pri, :α_du, :μ, :kiter],
      [Int, T, T, T, T, T, T, T, T, T, T, T, Int],
      hdr_override = Dict(
        :k => "iter",
        :pri_obj => "obj",
        :pdd => "rgap",
        :rbNorm => "‖rb‖",
        :rcNorm => "‖rc‖",
      ),
    )
    @info log_row(
      Any[
        cnts.k,
        itd.minimize ? itd.pri_obj : -itd.pri_obj,
        itd.pdd,
        res.rbNorm,
        res.rcNorm,
        zero(T),
        zero(T),
        itd.μ,
        get_kiter(pad),
      ],
    )
  end

  if iconf.mode == :multi
    # iter in Float32 then convert data to Float64
    pt, itd, res, dda, pad = iter_and_update_T!(
      pt,
      itd,
      fd32,
      id,
      res,
      sc,
      dda,
      pad,
      ϵ32,
      ϵ,
      cnts,
      itol.max_iter32,
      Float64,
      display,
    )

    if T0 == Float128
      # iters in Float64 then convert data to Float128
      pt, itd, res, dda, pad = iter_and_update_T!(
        pt,
        itd,
        fd64,
        id,
        res,
        sc,
        dda,
        pad,
        ϵ64,
        ϵ,
        cnts,
        itol.max_iter64,
        Float128,
        display,
      )
    end
    sc.max_iter = itol.max_iter
  end

  ## iter T0
  # refinement
  if !sc.optimal
    if iconf.refinement == :zoom || iconf.refinement == :ref
      ϵz = Tolerances(
        T(1),
        T(itol.ϵ_rbz),
        T(itol.ϵ_rbz),
        T(ϵ.tol_rb * T(itol.ϵ_rbz / itol.ϵ_rb)),
        one(T),
        T(itol.ϵ_μ),
        T(itol.ϵ_Δx),
        iconf.normalize_rtol,
      )
      iter!(pt, itd, fd_T0, id, res, sc, dda, pad, ϵz, cnts, T0, display)
      sc.optimal = false

      fd_ref, pt_ref = fd_refinement(
        fd_T0,
        id,
        res,
        itd.Δxy,
        pt,
        itd,
        ϵ,
        dda,
        pad,
        spd,
        cnts,
        T0,
        iconf.refinement,
      )
      iter!(pt_ref, itd, fd_ref, id, res, sc, dda, pad, ϵ, cnts, T0, display)
      update_pt_ref!(fd_ref.Δref, pt, pt_ref, res, id, fd_T0, itd)

    elseif iconf.refinement == :multizoom || iconf.refinement == :multiref
      spd = convert(StartingPointData{T0, typeof(pt.x)}, spd)
      fd_ref, pt_ref = fd_refinement(
        fd_T0,
        id,
        res,
        itd.Δxy,
        pt,
        itd,
        ϵ,
        dda,
        pad,
        spd,
        cnts,
        T0,
        iconf.refinement,
        centering = true,
      )
      iter!(pt_ref, itd, fd_ref, id, res, sc, dda, pad, ϵ, cnts, T0, display)
      update_pt_ref!(fd_ref.Δref, pt, pt_ref, res, id, fd_T0, itd)

    else
      # iters T0, no refinement
      iter!(pt, itd, fd_T0, id, res, sc, dda, pad, ϵ, cnts, T0, display)
    end
  end

  if iconf.scaling
    post_scale!(sd.d1, sd.d2, sd.d3, pt, res, fd_T0, id, itd)
  end

  if cnts.k >= itol.max_iter
    status = :max_iter
  elseif sc.tired
    status = :max_time
  elseif sc.optimal
    status = :acceptable
  else
    status = :unknown
  end

  if iconf.presolve
    x = similar(QM0.meta.x0)
    postsolve!(QM0, QM, pt.x, x)
    nrm = length(QM.xrm)
  else
    x = pt.x[1:(idi.nvar)]
    nrm = 0
  end

  multipliers, multipliers_L, multipliers_U =
    get_multipliers(pt.s_l, pt.s_u, id.ilow, id.iupp, id.nvar, pt.y, idi, nrm)

  if typeof(res) <: ResidualsHistory
    solver_specific = Dict(
      :absolute_iter_cnt => cnts.k,
      :pdd => itd.pdd,
      :rbNormH => res.rbNormH,
      :rcNormH => res.rcNormH,
      :pddH => res.pddH,
      :nprodH => res.kiterH,
      :μH => res.μH,
      :min_bound_distH => res.min_bound_distH,
      :KresNormH => res.KresNormH,
      :KresPNormH => res.KresPNormH,
      :KresDNormH => res.KresDNormH,
    )
  else
    solver_specific = Dict(:absolute_iter_cnt => cnts.k, :pdd => itd.pdd)
  end

  elapsed_time = time() - sc.start_time

  stats = GenericExecutionStats(
    status,
    QM,
    solution = x,
    objective = itd.minimize ? itd.pri_obj : -itd.pri_obj,
    dual_feas = res.rcNorm,
    primal_feas = res.rbNorm,
    multipliers = multipliers,
    multipliers_L = multipliers_L,
    multipliers_U = multipliers_U,
    iter = cnts.km,
    elapsed_time = elapsed_time,
    solver_specific = solver_specific,
  )
  return stats
end

function ripqp(LLS::LLSModel; iconf::InputConfig{Int} = InputConfig(), kwargs...)
  iconf.sp.δ0 = 0.0 # equality constraints of least squares as QPs are already regularized
  FLLS = FeasibilityFormNLS(LLS)
  stats = ripqp(QuadraticModel(FLLS, FLLS.meta.x0, name = LLS.meta.name); iconf = iconf, kwargs...)
  n = LLS.meta.nvar
  x, r = stats.solution[1:n], stats.solution[(n + 1):end]
  solver_sp = stats.solver_specific
  solver_sp[:r] = r
  return GenericExecutionStats(
    stats.status,
    LLS,
    solution = x,
    objective = stats.objective,
    dual_feas = stats.dual_feas,
    primal_feas = stats.primal_feas,
    multipliers = stats.multipliers,
    multipliers_L = stats.multipliers_L,
    multipliers_U = stats.multipliers_U,
    iter = stats.iter,
    elapsed_time = stats.elapsed_time,
    solver_specific = solver_sp,
  )
end

end
