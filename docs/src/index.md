```@meta
CurrentModule = MaximumWeightTwoStageSpanningTree
```

# MaximumWeightTwoStageSpanningTree.jl

Documentation for [MaximumWeightTwoStageSpanningTree](https://github.com/axelparmentier/MaximumWeightTwoStageSpanningTree.jl).

This package enables to reproduce the numerical experiments in [Learning structured approximations of operations research problems](https://hal.science/hal-03281894).

Further details on the problem considered are available [here](problem.md).

Details on the Lagrangian relaxation and the Lagrangian heuristic algorithms used in the paper experiments are available [here](optimization.html).

### Numerical experiments in the results.

A notebook with the numerical experiments run in the paper is available [here](run_paper_experiments.html).

In order to reproduce the paper experiments, open a `julia` repl in the package folder, and run the following code.

```
    using Pkg
    Pkg.activate(".")
    Pkg.activate("./scripts")
    Pkg.instantiate()
    include("scripts/run_paper_experiments.jl")
```
