using Circuitscape
using LinearAlgebra
BLAS.set_num_threads(1)
jobs = readlines(ARGS[1])
println("Circuitscape version ", Base.pkgversion(Circuitscape))
for (i, config) in enumerate(jobs)
    println("JOB ",i,"/",length(jobs)," ",config); flush(stdout)
    elapsed = @elapsed compute(config)
    println("DONE ",i," elapsed_seconds=",elapsed); flush(stdout)
end
println("ALL_CIRCUIT_JOBS_COMPLETE")
