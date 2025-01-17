mutable struct QM_FloatData_ref{T <: Real, S, M1, M2} <: Abstract_QM_FloatData{T, S, M1, M2}
  Q::M1
  A::M2 # using AT is easier to form systems
  b::S
  c::S
  c0::T
  lvar::S
  uvar::S
  Δref::T
  x_approx::S
  x_approxQx_approx_2::T
  b_init::S
  bTy_approx::T
  c_init::S
  pri_obj_approx::T
  uplo::Symbol
end

function fd_refinement(
  fd::QM_FloatData{T},
  id::QM_IntData,
  res::AbstractResiduals{T},
  Δxy::Vector{T},
  pt::Point{T},
  itd::IterData{T},
  ϵ::Tolerances{T},
  dda::DescentDirectionAllocs{T},
  pad::PreallocatedData{T},
  spd::StartingPointData{T},
  cnts::Counters,
  T0::DataType,
  mode::Symbol;
  centering::Bool = false,
) where {T <: Real}

  # center Points before zoom
  if centering
    # if pad.fact_fail # re-factorize if it failed in a lower precision system
    #     out = factorize_K2!(pad.K, pad.K_fact, pad.D, pad.diag_Q, pad.diagind_K, pad.regu, 
    #                         pt.s_l, pt.s_u, itd.x_m_lvar, itd.uvar_m_x, id.ilow, id.iupp, 
    #                         id.ncon, id.nvar, cnts, itd.qp, T, T0)
    # end
    # dda.rxs_l .= -itd.μ 
    # dda.rxs_u .= itd.μ 
    # solve_augmented_system_cc!(pad.K_fact, itd.Δxy, itd.Δs_l, itd.Δs_u, itd.x_m_lvar, itd.uvar_m_x, 
    #                         pad.rxs_l, pad.rxs_u, pt.s_l, pt.s_u, id.ilow, id.iupp)
    itd.Δxy .= 0
    itd.Δxy[id.ilow] .-= itd.μ ./ itd.x_m_lvar
    itd.Δxy[id.iupp] .+= itd.μ ./ itd.uvar_m_x
    padT = typeof(pad)
    if (padT <: PreallocatedDataAugmentedLDL && !factorized(pad.K_fact)) || (
      padT <: PreallocatedDataK2Krylov &&
      typeof(pad.pdat) <: LDLData &&
      !factorized(pad.pdat.K_fact)
    )
      out = update_pad!(pad, dda, pt, itd, fd, id, res, cnts, T0)
      out = solver!(itd.Δxy, pad, dda, pt, itd, fd, id, res, cnts, T0, :IPF)
    else
      out = solver!(itd.Δxy, pad, dda, pt, itd, fd, id, res, cnts, T0, :cc)
    end
    itd.Δs_l .= @views .-(.-itd.μ .+ pt.s_l .* itd.Δxy[id.ilow]) ./ itd.x_m_lvar
    itd.Δs_u .= @views (itd.μ .+ pt.s_u .* itd.Δxy[id.iupp]) ./ itd.uvar_m_x

    α_pri, α_dual =
      compute_αs(pt.x, pt.s_l, pt.s_u, fd.lvar, fd.uvar, itd.Δxy, itd.Δs_l, itd.Δs_u, id.nvar)
    update_data!(pt, α_pri, α_dual, itd, pad, res, fd, id)
  end

  c_ref = fd.c .- itd.ATy .+ itd.Qx
  αref = T(1.0e12)
  if mode == :zoom || mode == :multizoom
    # zoom parameter
    Δref = one(T) / res.rbNorm
  elseif mode == :ref || mode == :multiref
    Δref = one(T)
    δd = norm(c_ref, Inf)
    if id.nlow == 0 && id.nupp > 0
      δp = max(res.rbNorm, minimum(itd.uvar_m_x))
    elseif id.nlow > 0 && id.nupp == 0
      δp = max(res.rbNorm, minimum(itd.x_m_lvar))
    elseif id.nlow == 0 && id.nupp == 0
      δp = max(res.rbNorm)
    else
      δp = max(res.rbNorm, minimum(itd.x_m_lvar), minimum(itd.uvar_m_x))
    end
    Δref = min(αref / Δref, one(T) / δp, one(T) / δd)
  end
  if Δref == T(Inf)
    Δref = αref
  end

  c_ref .*= Δref
  fd_ref = QM_FloatData_ref(
    fd.Q,
    fd.A,
    .-res.rb .* Δref,
    c_ref,
    fd.c0,
    Δref .* (fd.lvar .- pt.x),
    Δref .* (fd.uvar .- pt.x),
    Δref,
    pt.x,
    copy(itd.xTQx_2),
    fd.b,
    dot(fd.b, pt.y),
    fd.c,
    copy(itd.pri_obj),
    fd.uplo,
  )

  # init zoom Point
  S = typeof(fd.c)
  pt_z = Point(
    fill!(S(undef, id.nvar), zero(T)),
    fill!(S(undef, id.ncon), zero(T)),
    pt.s_l .* Δref,
    pt.s_u .* Δref,
  )
  mul!(itd.Qx, fd_ref.Q, pt_z.x)
  fd_ref.uplo == :U ? mul!(itd.ATy, fd_ref.A, pt_z.y) : mul!(itd.ATy, fd_ref.A', pt_z.y)
  itd.x_m_lvar .= @views pt_z.x[id.ilow] .- fd_ref.lvar[id.ilow]
  itd.uvar_m_x .= @views fd_ref.uvar[id.iupp] .- pt_z.x[id.iupp]

  # update residuals
  res.rb .= itd.Ax .- fd_ref.b
  res.rc .= itd.ATy .- itd.Qx .- fd_ref.c
  res.rc[id.ilow] .+= pt_z.s_l
  res.rc[id.iupp] .-= pt_z.s_u
  res.rcNorm, res.rbNorm = norm(res.rc, Inf), norm(res.rb, Inf)

  return fd_ref, pt_z
end

function update_pt_ref!(
  Δref::T,
  pt::Point{T},
  pt_z::Point{T},
  res::AbstractResiduals{T},
  id::QM_IntData,
  fd::QM_FloatData{T},
  itd::IterData{T},
) where {T <: Real}

  # update Point                    
  pt.x .+= pt_z.x ./ Δref
  pt.y .+= pt_z.y ./ Δref
  pt.s_l .= pt_z.s_l ./ Δref
  pt.s_u .= pt_z.s_u ./ Δref

  # update IterData
  itd.Qx = mul!(itd.Qx, fd.Q, pt.x)
  itd.xTQx_2 = dot(pt.x, itd.Qx) / 2
  fd.uplo == :U ? mul!(itd.ATy, fd.A, pt.y) : mul!(itd.ATy, fd.A', pt.y)
  fd.uplo == :U ? mul!(itd.Ax, fd.A', pt.x) : mul!(itd.Ax, fd.A, pt.x)
  itd.cTx = dot(fd.c, pt.x)
  itd.pri_obj = itd.xTQx_2 + itd.cTx + fd.c0
  itd.dual_obj = @views dot(fd.b, pt.y) - itd.xTQx_2 + dot(pt.s_l, fd.lvar[id.ilow]) -
         dot(pt.s_u, fd.uvar[id.iupp]) + fd.c0
  itd.pdd = abs(itd.pri_obj - itd.dual_obj) / (one(T) + abs(itd.pri_obj))

  # update Residuals
  res.rb .= itd.Ax .- fd.b
  res.rc .= itd.ATy .- itd.Qx .- fd.c
  res.rc[id.ilow] .+= pt.s_l
  res.rc[id.iupp] .-= pt.s_u
  res.rcNorm, res.rbNorm = norm(res.rc, Inf), norm(res.rb, Inf)
  res.rcNorm, res.rbNorm = norm(res.rc, Inf), norm(res.rb, Inf)
end

function update_data!(
  pt::Point{T},
  α_pri::T,
  α_dual::T,
  itd::IterData{T},
  pad::PreallocatedData{T},
  res::AbstractResiduals{T},
  fd::QM_FloatData_ref{T},
  id::QM_IntData,
) where {T <: Real}

  # (x, y, s_l, s_u) += α * Δ
  update_pt!(
    pt.x,
    pt.y,
    pt.s_l,
    pt.s_u,
    α_pri,
    α_dual,
    itd.Δxy,
    itd.Δs_l,
    itd.Δs_u,
    id.ncon,
    id.nvar,
  )

  # update IterData
  itd.x_m_lvar .= @views pt.x[id.ilow] .- fd.lvar[id.ilow]
  itd.uvar_m_x .= @views fd.uvar[id.iupp] .- pt.x[id.iupp]
  boundary_safety!(itd.x_m_lvar, itd.uvar_m_x)
  itd.μ = compute_μ(itd.x_m_lvar, itd.uvar_m_x, pt.s_l, pt.s_u, id.nlow, id.nupp)

  if itd.perturb && itd.μ ≤ eps(T)
    perturb_x!(
      pt.x,
      pt.s_l,
      pt.s_u,
      itd.x_m_lvar,
      itd.uvar_m_x,
      fd.lvar,
      fd.uvar,
      itd.μ,
      id.ilow,
      id.iupp,
      id.nlow,
      id.nupp,
      id.nvar,
    )
    itd.μ = compute_μ(itd.x_m_lvar, itd.uvar_m_x, pt.s_l, pt.s_u, id.nlow, id.nupp)
  end

  itd.Qx = mul!(itd.Qx, fd.Q, pt.x)
  x_approxQx = dot(fd.x_approx, itd.Qx)
  itd.xTQx_2 = dot(pt.x, itd.Qx) / 2
  fd.uplo == :U ? mul!(itd.ATy, fd.A, pt.y) : mul!(itd.ATy, fd.A', pt.y)
  fd.uplo == :U ? mul!(itd.Ax, fd.A', pt.x) : mul!(itd.Ax, fd.A, pt.x)
  itd.cTx = dot(fd.c_init, pt.x)
  itd.pri_obj =
    fd.pri_obj_approx +
    x_approxQx / fd.Δref +
    itd.xTQx_2 / fd.Δref^2 +
    dot(fd.c_init, pt.x) / fd.Δref
  itd.dual_obj = @views fd.bTy_approx + dot(fd.b_init, pt.y) / fd.Δref - fd.x_approxQx_approx_2 -
         dot(fd.x_approx, itd.Qx) / fd.Δref - itd.xTQx_2 / fd.Δref^2 +
         dot(pt.s_l, fd.lvar[id.ilow]) / fd.Δref^2 +
         dot(pt.s_l, fd.x_approx[id.ilow]) / fd.Δref - dot(pt.s_u, fd.uvar[id.iupp]) / fd.Δref^2 -
         dot(pt.s_u, fd.x_approx[id.iupp]) / fd.Δref + fd.c0
  itd.pdd = abs(itd.pri_obj - itd.dual_obj) / (one(T) + abs(itd.pri_obj))

  #update Residuals
  res.rb .= itd.Ax .- fd.b
  res.rc .= itd.ATy .- itd.Qx .- fd.c
  res.rc[id.ilow] .+= pt.s_l
  res.rc[id.iupp] .-= pt.s_u
  res.rcNorm, res.rbNorm = norm(res.rc, Inf) / fd.Δref, norm(res.rb, Inf) / fd.Δref
end
