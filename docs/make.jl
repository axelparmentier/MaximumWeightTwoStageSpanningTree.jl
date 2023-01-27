using MaximumWeightTwoStageSpanningTree
using Documenter
using Literate

experiments_files = [
    "run_paper_experiments.jl",
]

for exp_file in experiments_files
    exp_file = joinpath(dirname(@__DIR__), "scripts", exp_file)
    exp_md_dir = joinpath(@__DIR__, "src")
    Literate.markdown(
        exp_file, exp_md_dir; documenter=true, execute=false, codefence="```julia" => "```"
    )
end

DocMeta.setdocmeta!(
    MaximumWeightTwoStageSpanningTree, :DocTestSetup, :(using MaximumWeightTwoStageSpanningTree); recursive=true
)

makedocs(;
    modules=[MaximumWeightTwoStageSpanningTree],
    authors="Axel Parmentier and contributors",
    repo="https://github.com/axelparmentier/MaximumWeightTwoStageSpanningTree.jl/blob/{commit}{path}#{line}",
    sitename="MaximumWeightTwoStageSpanningTree.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://axelparmentier.github.io/MaximumWeightTwoStageSpanningTree.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
        "Problem statement" => "problem.md",
        "Optimization algorithms" => "optimization.md",
        "Experiments" => "run_paper_experiments.md",
        "API reference" => "api.md",
    ],
)

deploydocs(; repo="github.com/axelparmentier/MaximumWeightTwoStageSpanningTree.jl", devbranch="main")
