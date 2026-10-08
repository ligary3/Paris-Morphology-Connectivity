using Circuitscape

isempty(ARGS) && error("Pass one or more Circuitscape ini files")
for config in ARGS
    println("Running: ", config)
    compute(config)
end
println("Completed ", length(ARGS), " matched Circuit runs")
