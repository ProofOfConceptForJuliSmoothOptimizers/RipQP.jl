# conversion function if QM.data.H and QM.data.A are not in the type required by iconf.sp
function convert_QM(
  QM::QuadraticModel{T, S, M1, M2},
  iconf::InputConfig,
  display::Bool,
) where {T, S, M1, M2}
  T_sp = typeof(iconf.sp)
  if (T_sp <: K2LDLParams || T_sp <: K2_5LDLParams) &&
     !(M1 <: SparseMatrixCOO) &&
     !(M2 <: SparseMatrixCOO)
    QM = convert(QuadraticModel{T, S, SparseMatrixCOO{T, Int}, SparseMatrixCOO{T, Int}}, QM)
  end
  # deactivate presolve and scaling if H and A are not SparseMatricesCOO
  # (TODO: write scaling and presolve for other types)
  M12, M22 = typeof(QM.data.H), typeof(QM.data.A)
  if !(M12 <: SparseMatrixCOO) || !(M22 <: SparseMatrixCOO)
    display &&
      iconf.presolve &&
      @warn "No presolve operations available if QM.data.H and QM.data.A are not SparseMatricesCOO"
    iconf.presolve = false
  end
  if M12 <: AbstractLinearOperator || M22 <: AbstractLinearOperator
    display &&
      iconf.scaling &&
      @warn "No scaling operations available if QM.data.H and QM.data.A are LinearOperators"
    iconf.scaling = false
  end
  return QM
end

function vcatsort(v1, v2)
  n2 = length(v2)
  n2 == 0 && return v1
  n1 = length(v1)
  n1 == 0 && return v2

  n = n1 + n2
  res = similar(v1, n)
  c1, c2 = 1, 1
  @inbounds for i = 1:n
    if c2 == n2 + 1
      res[i] = v1[c1]
      c1 += 1
    elseif c1 == n1 + 1
      res[i] = v2[c2]
      c2 += 1
    else
      if v1[c1] < v2[c2]
        res[i] = v1[c1]
        c1 += 1
      else
        res[i] = v2[c2]
        c2 += 1
      end
    end
  end

  return res
end

function sparse_dropzeros(rows, cols, vals::Vector, nrows, ncols)
  M = sparse(rows, cols, vals, ncols, nrows)
  dropzeros!(M)
  return M
end

function get_mat_QPData(
  A::SparseMatrixCOO{T, Int},
  H::SparseMatrixCOO{T, Int},
  nvar::Int,
  ncon::Int,
  uplo::Symbol,
) where {T}
  if uplo == :U # A is Aᵀ of QuadraticModel QM
    fdA = sparse_dropzeros(A.cols, A.rows, A.vals, ncon, nvar)
    fdQ = sparse_dropzeros(H.cols, H.rows, H.vals, nvar, nvar)
  else
    fdA = sparse_dropzeros(A.rows, A.cols, A.vals, nvar, ncon)
    fdQ = sparse_dropzeros(H.rows, H.cols, H.vals, nvar, nvar)
  end
  return fdA, Symmetric(fdQ, uplo)
end

function get_mat_QPData(A, H, nvar::Int, ncon::Int, uplo::Symbol)
  fdA = uplo == :U ? transpose(A) : A
  return fdA, Symmetric(H, :L)
end

function switch_H_to_max!(
  data::QuadraticModels.QPData{T, S, M1, M2},
) where {T, S, M1 <: SparseMatrixCOO, M2 <: SparseMatrixCOO}
  data.H.vals .= .-data.H.vals
end

function switch_H_to_max!(data::QuadraticModels.QPData)
  data.H = -data.H
end

function get_QM_data(QM::AbstractQuadraticModel{T, S}, uplo::Symbol) where {T <: Real, S}
  # constructs A and Q transposed so we can create K upper triangular. 
  A, Q = get_mat_QPData(QM.data.A, QM.data.H, QM.meta.nvar, QM.meta.ncon, uplo)
  id = QM_IntData(
    vcatsort(QM.meta.ilow, QM.meta.irng),
    vcatsort(QM.meta.iupp, QM.meta.irng),
    QM.meta.irng,
    QM.meta.ifree,
    QM.meta.ifix,
    QM.meta.ncon,
    QM.meta.nvar,
    0,
    0,
  )
  id.nlow, id.nupp = length(id.ilow), length(id.iupp) # number of finite constraints
  @assert QM.meta.lcon == QM.meta.ucon # equality constraint (Ax=b)
  @assert length(QM.meta.lvar) == length(QM.meta.uvar) == QM.meta.nvar
  fd = QM_FloatData(Q, A, QM.meta.lcon, QM.data.c, QM.data.c0, QM.meta.lvar, QM.meta.uvar, uplo)
  return fd, id
end

function allocate_workspace(
  QM::AbstractQuadraticModel,
  iconf::InputConfig,
  itol::InputTol,
  start_time,
  T0::DataType,
)
  sc = StopCrit(false, false, false, itol.max_iter, itol.max_time, start_time, 0.0)

  # save inital IntData to compute multipliers at the end of the algorithm
  idi = IntDataInit(
    QM.meta.nvar,
    QM.meta.ncon,
    QM.meta.ilow,
    QM.meta.iupp,
    QM.meta.irng,
    QM.meta.ifix,
    QM.meta.jlow,
    QM.meta.jupp,
    QM.meta.jrng,
    QM.meta.jfix,
  )

  if !QM.meta.minimize
    switch_H_to_max!(QM.data)
    QM.data.c .= .-QM.data.c
    QM.data.c0 = -QM.data.c0
  end

  QM = SlackModel(QM)

  if QM.meta.ncon == length(QM.meta.jfix) && !iconf.presolve && iconf.scaling
    QM = deepcopy(QM) # if not modified by SlackModel and presolve
  end

  uplo = iconf.sp.uplo
  fd_T0, id = get_QM_data(QM, uplo) # apply presolve at the same time

  T = T0 # T0 is the data type, in mode :multi T will gradually increase to T0
  ϵ = Tolerances(
    T(itol.ϵ_pdd),
    T(itol.ϵ_rb),
    T(itol.ϵ_rc),
    one(T),
    one(T),
    T(itol.ϵ_μ),
    T(itol.ϵ_Δx),
    iconf.normalize_rtol,
  )

  sd = ScaleData(fd_T0, id, iconf.scaling)

  # allocate data for iterations
  if iconf.mode == :multi || iconf.mode == :multiref || iconf.mode == :multizoom
    T = iconf.Timulti
  end
  S0 = typeof(fd_T0.c)
  S = change_vector_eltype(S0, T)

  res = init_residuals(S(undef, id.ncon), S(undef, id.nvar), zero(T), zero(T), iconf, id)

  itd = IterData(
    S(undef, id.nvar + id.ncon), # Δxy
    S(undef, id.nlow), # Δs_l
    S(undef, id.nupp), # Δs_u
    S(undef, id.nlow), # x_m_lvar
    S(undef, id.nupp), # uvar_m_x
    S(undef, id.nvar), # init Qx
    S(undef, id.nvar), # init ATy
    S(undef, id.ncon), # Ax
    zero(T), #xTQx
    zero(T), #cTx
    zero(T), #pri_obj
    zero(T), #dual_obj
    zero(T), #μ
    zero(T),#pdd
    zeros(T, 6), #l_pdd
    one(T), #mean_pdd
    typeof(fd_T0.Q) <: Union{AbstractLinearOperator, DenseMatrix} || nnz(fd_T0.Q.data) > 0,
    QM.meta.minimize,
    iconf.perturb,
  )

  dda_type = Symbol(:DescentDirectionAllocs, iconf.solve_method)
  dda = DescentDirectionAllocs(id, iconf.solve_method, S)

  cnts = Counters(0, 0, 0, 0, iconf.kc, 0, iconf.w)

  pt = Point(S(undef, id.nvar), S(undef, id.ncon), S(undef, id.nlow), S(undef, id.nupp))

  spd = StartingPointData{T, S}(S(undef, id.nvar), S(undef, id.nlow), S(undef, id.nupp))

  #####
  return sc, idi, fd_T0, id, ϵ, res, itd, dda, pt, sd, spd, cnts, T
end

function allocate_extra_workspace_32(itol::InputTol, iconf::InputConfig, fd_T0::QM_FloatData)
  T = Float32
  ϵ32 = Tolerances(
    T(itol.ϵ_pdd32),
    T(itol.ϵ_rb32),
    T(itol.ϵ_rc32),
    one(T),
    one(T),
    T(itol.ϵ_μ),
    T(itol.ϵ_Δx),
    iconf.normalize_rtol,
  )
  fd32 = convert_FloatData(T, fd_T0)
  return fd32, ϵ32
end

function allocate_extra_workspace_64(itol::InputTol, iconf::InputConfig, fd_T0::QM_FloatData)
  T = Float64
  fd64 = convert_FloatData(T, fd_T0)
  ϵ64 = Tolerances(
    T(itol.ϵ_pdd64),
    T(itol.ϵ_rb64),
    T(itol.ϵ_rc64),
    one(T),
    one(T),
    T(itol.ϵ_μ),
    T(itol.ϵ_Δx),
    iconf.normalize_rtol,
  )
  return fd64, ϵ64
end

function initialize!(
  fd::QM_FloatData{T},
  id::QM_IntData,
  res::AbstractResiduals{T},
  itd::IterData{T},
  dda::DescentDirectionAllocs{T},
  pt::Point{T},
  spd::StartingPointData{T},
  ϵ::Tolerances{T},
  sc::StopCrit{Tc},
  iconf::InputConfig{Tconf},
  cnts::Counters,
  T0::DataType,
) where {T <: Real, Tc <: Real, Tconf <: Real}
  # init system
  # solve [-Q-D    A' ] [x] = [0]  to initialize (x, y, s_l, s_u)
  #       [  A     0  ] [y] = [b]
  itd.Δxy[1:(id.nvar)] .= 0
  itd.Δxy[(id.nvar + 1):end] = fd.b
  if typeof(iconf.sp) <: NewtonParams
    itd.Δs_l .= zero(T)
    itd.Δs_u .= zero(T)
    pt.s_l .= one(T)
    pt.s_u .= one(T)
    itd.x_m_lvar .= one(T)
    itd.uvar_m_x .= one(T)
  end
  @timeit_debug to "init solver" begin
    pad = PreallocatedData(iconf.sp, fd, id, itd, pt, iconf)
    out = solver!(itd.Δxy, pad, dda, pt, itd, fd, id, res, cnts, T0, :init)
  end
  pt.x .= itd.Δxy[1:(id.nvar)]
  pt.y .= itd.Δxy[(id.nvar + 1):(id.nvar + id.ncon)]

  starting_points!(pt, fd, id, itd, spd)

  # stopping criterion
  #     rcNorm, rbNorm = norm(rc), norm(rb)
  #     optimal = pdd < ϵ_pdd && rbNorm < ϵ_rb && rcNorm < ϵ_rc
  res.rb .= itd.Ax .- fd.b
  res.rc .= itd.ATy .- itd.Qx .- fd.c
  res.rc[id.ilow] .+= pt.s_l
  res.rc[id.iupp] .-= pt.s_u
  res.rcNorm, res.rbNorm = norm(res.rc, Inf), norm(res.rb, Inf)
  typeof(res) <: ResidualsHistory && push_history_residuals!(res, itd, pad, id)
  set_tol_residuals!(ϵ, res.rbNorm, res.rcNorm)

  sc.optimal = itd.pdd < ϵ.pdd && res.rbNorm < ϵ.tol_rb && res.rcNorm < ϵ.tol_rc
  sc.small_μ = itd.μ < ϵ.μ

  return pad
end

function set_tol_residuals!(ϵ::Tolerances{T}, rbNorm::T, rcNorm::T) where {T <: Real}
  if ϵ.normalize_rtol == true
    ϵ.tol_rb, ϵ.tol_rc = ϵ.rb * (one(T) + rbNorm), ϵ.rc * (one(T) + rcNorm)
  else
    ϵ.tol_rb, ϵ.tol_rc = ϵ.rb, ϵ.rc
  end
end

############ tools for sparse matrices ##############

function get_diag_sparseCSC(M_colptr, n; tri = :U)
  # get diagonal index of M.nzval
  # we assume all columns of M are non empty, and M triangular (:L or :U)
  @assert tri == :U || tri == :L
  if tri == :U
    diagind = M_colptr[2:end] .- one(Int)
  else
    diagind = M_colptr[1:(end - 1)]
  end
  return diagind
end

function get_diag_Q(Q_colptr, Q_rowval, Q_nzval::Vector{T}, n) where {T <: Real}
  diagval = spzeros(T, n)
  @inbounds @simd for j = 1:n
    for k = Q_colptr[j]:(Q_colptr[j + 1] - 1)
      if j == Q_rowval[k]
        diagval[j] = Q_nzval[k]
      end
    end
  end
  return diagval
end

function get_diag_Q_dense(Q::SparseMatrixCSC{T, Int}, uplo::Symbol) where {T <: Real}
  n = size(Q, 1)
  diagval = zeros(T, n)
  fill_diag_Q_dense!(Q.colptr, Q.rowval, Q.nzval, diagval, n, uplo)
  return diagval
end

get_diag_Q_dense(Q::Symmetric{T, SparseMatrixCSC{T, Int}}, uplo::Symbol) where {T} =
  get_diag_Q_dense(Q.data, uplo)

function fill_diag_Q_dense!(
  Q_colptr,
  Q_rowval,
  Q_nzval::Vector{T},
  diagval::Vector{T},
  n,
  uplo::Symbol,
) where {T <: Real}
  for j = 1:n
    if uplo == :U
      k = Q_colptr[j + 1] - 1
      if k > 0
        i = Q_rowval[k]
        if j == i
          diagval[j] = Q_nzval[k]
        end
      end
    elseif uplo == :L
      k = Q_colptr[j]
      nnzQ = length(Q_nzval)
      if k ≤ nnzQ
        i = Q_rowval[k]
        if j == i
          diagval[j] = Q_nzval[k]
        end
      end
    end
  end
end
