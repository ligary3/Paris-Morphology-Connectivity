using Circuitscape

if length(ARGS) != 1
    error("Usage: julia run_circuitscape_config.jl <config.ini>")
end

println("Running Circuitscape config: ", ARGS[1])
result = compute(ARGS[1])
println("Circuitscape completed; return type: ", typeof(result))
