# TwoStageSpanningTree

Companion code to the paper [Learning structured approximations of operations research problems](https://hal.science/hal-03281894).

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
    Pkg.instantiate()
    include("scripts/run_paper_experiments.jl")
```