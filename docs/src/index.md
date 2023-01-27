```@meta
CurrentModule = MaximumWeightTwoStageSpanningTree
```

# TwoStageSpanningTree

Documentation for [MaximumWeightTwoStageSpanningTree](https://github.com/axelparmentier/MaximumWeightTwoStageSpanningTree.jl).

This package enables to 

## More details on the problem considered in the paper

Details on the Lagrangian relaxation and the Lagrangian heuristic algorithms used in the paper experiments are available on page [optimization](optimization.html)


###

In order to reproduce the paper experiments

Open a `julia` repl in the package folder, and run the following code

```
    using Pkg
    Pkg.activate(".")
    Pkg.activate("./scripts")
    include("scripts/run_paper_experiments.jl")
```
