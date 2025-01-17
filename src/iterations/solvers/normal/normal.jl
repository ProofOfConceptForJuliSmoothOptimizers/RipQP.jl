abstract type NormalParams <: SolverParams end

abstract type NormalKrylovParams{PT <: AbstractPreconditioner} <: NormalParams end

abstract type PreallocatedDataNormal{T <: Real, S} <: PreallocatedData{T, S} end

abstract type PreallocatedDataNormalChol{T <: Real, S} <: PreallocatedDataNormal{T, S} end

uses_krylov(pad::PreallocatedDataNormalChol) = false

include("K1CholDense.jl")

abstract type PreallocatedDataNormalKrylov{T <: Real, S} <: PreallocatedDataNormal{T, S} end

uses_krylov(pad::PreallocatedDataAugmentedKrylov) = true

include("K1Krylov.jl")
include("K1_1Structured.jl")
include("K1_2Structured.jl")
