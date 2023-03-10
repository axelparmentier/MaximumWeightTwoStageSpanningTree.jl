# MaximumWeightTwoStageSpanningTree

Companion code to the paper [Learning structured approximations of operations research problems](https://hal.science/hal-03281894).

Detailed documentation [here](https://axelparmentier.github.io/MaximumWeightTwoStageSpanningTree.jl/dev/).

Code was tested only on Ubuntu 22.04 with Julia 1.8.2.

## Testing the package

Open a `julia` repl in the package folder, and run the following code

```
    using Pkg
    Pkg.activate(".")
    Pkg.test()
```

## Building the doc

Open a `julia` repl in the package folder, and run the following code

```
    using Pkg
    Pkg.activate(".")
    Pkg.activate("./docs")
    Pkg.rm("MaximumWeightTwoStageSpanningTree") # Manifest.toml is not pushed, it would take the registry version, which does not exist
    Pkg.develop(path=".") # Takes instead the local version
    Pkg.instantiate()
    include("docs/make.jl")
```

You can then open with a web browser the file `docs/build/index.html`

## Reproducing the paper numerical experiments

Open a `julia` repl in the package folder, and run the following code

```
    using Pkg
    Pkg.activate(".")
    Pkg.activate("./scripts")
    Pkg.rm("MaximumWeightTwoStageSpanningTree") # Manifest.toml is not pushed, it would take the registry version, which does not exist
    Pkg.develop(path=".") # Takes instead the local version
    Pkg.instantiate()
    include("scripts/run_paper_experiments.jl")
```