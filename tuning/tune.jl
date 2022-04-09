using Pkg
using Distributed
using SolverParameters
using SolverTuning
using SolverCore
using NLPModels
using BenchmarkTools
using QPSReader
using Logging
using JSON

const IS_LOAD_BALANCING = false
const PATH_PREFIX = IS_LOAD_BALANCING ? "" : "no_"
const BASE_FILE_PATH = joinpath(@__DIR__, "plots", "$(PATH_PREFIX)load_balancing")

# 1. Launch workers
init_workers(;nb_nodes=23, exec_flags="--project=$(@__DIR__)")

# Contains worker data at each iteration
V = Vector{Dict{Int, Float64}}
workers_data = Dict{Symbol, V}(:time => V(), :memory => V(), :failures => V())

# Contains nb of problems per worker at each iteration:
nb_problems_per_worker = Vector{Dict{Int, Vector{String}}}()

# 2. make modules available to all workers:
@everywhere begin
  using RipQP,
  SolverTuning,
  NLPModels,
  QuadraticModels
end

# 3. Setup problems
netlib_problem_path = [joinpath(fetch_netlib(), path) for path ∈ readdir(fetch_netlib()) if match(r".*\.(SIF|QPS)$", path) !== nothing]
# mm_problem_path = [joinpath(fetch_mm(), path) for path ∈ readdir(fetch_mm()) if match(r".*\.(SIF|QPS)$", path) !== nothing]

# problem_paths = cat(mm_problem_path, netlib_problem_path; dims=1)
problem_paths = netlib_problem_path

problems = QuadraticModel[]
for problem_path ∈ problem_paths
  try
    with_logger(NullLogger()) do
      qps = readqps(problem_path)
      p = QuadraticModel(qps)
      push!(problems, p)
    end
  catch e
    @warn e
  end
end

# Function that will count failures
function count_failures(bmark_results::Dict{Int, Float64}, stats_results::Dict{Int, AbstractExecutionStats})
  failure_penalty = 0.0
  for (pb_id, stats) in stats_results
    is_failure(stats) || continue
    failure_penalty += 25.0 * bmark_results[pb_id]
  end
  return failure_penalty
end

function is_failure(stats::AbstractExecutionStats)
  failure_status = [:exception, :infeasible, :max_eval, :max_iter, :max_time, :stalled, :neg_pred]
  return any(s -> s == stats.status, failure_status)
end

solver = RipQPSolver(first(problems))
kwargs = Dict{Symbol, Any}(:display => false)

function my_black_box(args...;kwargs...)
  # a little hack...
  bmark_results, stats_results, solver_results = eval_solver(ripqp, args...;kwargs...)
  bmark_results_time = Dict(pb_id => (median(bmark).time/1.0e9) for (pb_id, bmark) ∈ bmark_results)
  bmark_results_mem = Dict(pb_id => (median(bmark).memory/1.0e9) for (pb_id, bmark) ∈ bmark_results)

  total_time = sum(values(bmark_results_time))
  total_mem = sum(values(bmark_results_mem))
  failure_penalty = count_failures(bmark_results_time, stats_results)
  bb_result = total_time + total_mem + failure_penalty

  # Getting worker data:
  global workers_data
  worker_times = Dict(worker_id => 0.0 for worker_id in workers())
  worker_mems = Dict(worker_id => 0.0 for worker_id in workers())
  worker_failures = Dict(worker_id => 0.0 for worker_id in workers())
  for (worker_id, solver_result) in solver_results
    bmark_trials, stats = solver_result

    median_times = Dict(pb_id => (median(bmark).time/1.0e9) for (pb_id, bmark) ∈ bmark_trials)
    median_mem = Dict(pb_id => (median(bmark).memory/1.0e9) for (pb_id, bmark) ∈ bmark_trials)
    failure_pen = count_failures(median_times, stats)

    worker_times[worker_id] = isempty(values(median_times)) ? -1.0 : sum(values(median_times))
    worker_mems[worker_id] =  isempty(values(median_mem)) ? -1.0 : sum(values(median_mem))
    worker_failures[worker_id] = failure_pen
  end
  push!(workers_data[:time], worker_times)
  push!(workers_data[:memory], worker_mems)
  push!(workers_data[:failures], worker_failures)

  # Getting nb of problems per worker:
  global nb_problems_per_worker
  problems_per_worker = Dict{Int, Vector{String}}()
  for (worker_id, solver_result) in solver_results
    bmark_trials, _ = solver_result
    problems_per_worker[worker_id] = ["Problem#$(p_id)" for p_id in keys(bmark_trials)]
  end
  push!(nb_problems_per_worker, problems_per_worker)

  return [bb_result], bmark_results_time, stats_results
end
black_box = BlackBox(solver, collect(values(solver.parameters)), my_black_box, kwargs)

# 7. define problem suite
param_optimization_problem =
  ParameterOptimizationProblem(black_box, problems; is_load_balanced=IS_LOAD_BALANCING)
  # named arguments are options to pass to Nomad
create_nomad_problem!(
  param_optimization_problem;
  display_all_eval = true,
  # max_time = 18000,
  max_bb_eval =150,
  display_stats = ["BBE", "EVAL", "SOL", "OBJ"],
)

# 8. Execute Nomad
result = solve_with_nomad!(param_optimization_problem)
@info ("Best feasible parameters: $(result.x_best_feas)")

for p ∈ black_box.solver_params
  @info "$(name(p)): $(default(p))"
end

# discard the first iteration
plot_data = Dict{Symbol, Dict{Int, Vector{Float64}}}()
for key in keys(workers_data)
  plot_data[key] = Dict{Int, Vector{Float64}}()
  for worker_id in workers()
    plot_data[key][worker_id] = Float64[]
  end
end

for (metric_key, bb_iterations) in workers_data
  for bb_it in bb_iterations[2:end]
    for (worker_id, metric) in bb_it
      metric_dict = plot_data[metric_key]
      push!(metric_dict[worker_id], metric)
    end
  end
end

open(joinpath(BASE_FILE_PATH, "workers_data.json"), "w") do f
  JSON.print(f, plot_data)
end

open(joinpath(BASE_FILE_PATH, "workers_problems.json"), "w") do f
  JSON.print(f, nb_problems_per_worker)
end

open(joinpath(BASE_FILE_PATH, "final_time_per_problem.json"), "w") do f
  problem_weights = Dict(id => p.weight for (id, p) in param_optimization_problem.load_balancer.problems)
  JSON.print(f, problem_weights)
end

rmprocs(workers())
