@testset "dynamic_regularization" begin
  stats1 =
    ripqp(QuadraticModel(qps1), sp = K2LDLParams(regul = :dynamic), history = true, display = false)
  @test isapprox(stats1.objective, -1.59078179, atol = 1e-2)
  @test stats1.status == :first_order

  stats2 = ripqp(QuadraticModel(qps2), sp = K2LDLParams(regul = :dynamic), display = false)
  @test isapprox(stats2.objective, -9.99599999e1, atol = 1e-2)
  @test stats2.status == :first_order

  stats3 = ripqp(QuadraticModel(qps3), sp = K2LDLParams(regul = :dynamic), display = false)
  @test isapprox(stats3.objective, 5.32664756, atol = 1e-2)
  @test stats3.status == :first_order
end

@testset "centrality_corrections" begin
  stats1 = ripqp(QuadraticModel(qps1), kc = -1, display = false) # automatic centrality corrections computation
  @test isapprox(stats1.objective, -1.59078179, atol = 1e-2)
  @test stats1.status == :first_order

  stats2 = ripqp(QuadraticModel(qps2), kc = 2, display = false)
  @test isapprox(stats2.objective, -9.99599999e1, atol = 1e-2)
  @test stats2.status == :first_order

  stats3 = ripqp(QuadraticModel(qps3), kc = 2, display = false)
  @test isapprox(stats3.objective, 5.32664756, atol = 1e-2)
  @test stats3.status == :first_order
end

@testset "refinement" begin
  stats1 = ripqp(QuadraticModel(qps1), mode = :zoom, display = false) # automatic centrality corrections computation
  @test isapprox(stats1.objective, -1.59078179, atol = 1e-2)
  @test stats1.status == :first_order

  stats1 = ripqp(QuadraticModel(qps1), mode = :ref, display = false) # automatic centrality corrections computation
  @test isapprox(stats1.objective, -1.59078179, atol = 1e-2)
  @test stats1.status == :first_order

  stats2 = ripqp(QuadraticModel(qps2), mode = :multizoom, scaling = false, display = false)
  @test isapprox(stats2.objective, -9.99599999e1, atol = 1e-2)
  @test stats2.status == :first_order

  stats3 = ripqp(QuadraticModel(qps3), mode = :multiref, display = false)
  @test isapprox(stats3.objective, 5.32664756, atol = 1e-2)
  @test stats3.status == :first_order
end

@testset "IPF" begin
  stats1 = ripqp(QuadraticModel(qps1), display = false, solve_method = IPF(), mode = :multi)
  @test isapprox(stats1.objective, -1.59078179, atol = 1e-2)
  @test stats1.status == :first_order

  stats2 = ripqp(QuadraticModel(qps2), display = false, solve_method = IPF())
  @test isapprox(stats2.objective, -9.99599999e1, atol = 1e-2)
  @test stats2.status == :first_order

  stats2 = ripqp(
    QuadraticModel(qps2),
    display = false,
    solve_method = IPF(),
    sp = K2_5LDLParams(),
    mode = :zoom,
  )
  @test isapprox(stats2.objective, -9.99599999e1, atol = 1e-2)
  @test stats2.status == :first_order

  stats1 = ripqp(
    QuadraticModel(qps1),
    display = false,
    perturb = true,
    solve_method = IPF(),
    mode = :multiref,
  )
  @test isapprox(stats1.objective, -1.59078179, atol = 1e-2)
  @test stats1.status == :first_order

  stats2 = ripqp(
    QuadraticModel(qps2),
    display = false,
    solve_method = IPF(),
    mode = :multizoom,
    sp = K2_5LDLParams(),
  )
  @test isapprox(stats2.objective, -9.99599999e1, atol = 1e-2)
  @test stats2.status == :first_order

  stats2 = ripqp(
    QuadraticModel(qps2),
    display = false,
    solve_method = IPF(),
    mode = :zoom,
    sp = K2LDLParams(regul = :dynamic),
  )
  @test isapprox(stats2.objective, -9.99599999e1, atol = 1e-2)
  @test stats2.status == :first_order
end
