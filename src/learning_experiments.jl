"""
function compute_σ(X)

    Computes features standard deviation
"""
function compute_σ(X)
    nb_arcs = 0
    nb_features = size(X[1][1], 1)
    μ = zeros(nb_features, 1, 2)
    # σ = zeros(nb_features, 1, 2)
    σ = 0.0001 * ones(nb_features, 1, 2)

    for (x, _, (_, _, _, _)) in X
        nb_edges = size(x, 2)
        μ .+= sum(x; dims=2)
        nb_arcs += nb_edges
    end
    μ ./= nb_arcs

    for (x, _, (_, _, _, _)) in X
        σ .+= sum((x .- μ) .^ 2; dims=2)
    end
    σ ./= nb_arcs
    σ = sqrt.(σ)
    return σ
end

"""
function reduce_data!(X, σ)

    Standarizes features without centering them
"""
function reduce_data!(X, σ)
    for (x, _, (_, _, _, _)) in X
        for col in axes(x, 2) #  1:size(x, 2)
            x[:, col, :] = @views(x[:, col, :]) ./ dropdims(σ; dims=2)
        end
    end
end

"""
function build_dataset(;
    nb_scenarios=5:5:20,
    first_max=20:20,
    second_max=10:5:30,
    seeds=1:5,
    grid_sizes=4:6,
    solver=benders_solver,
)

    Build a training/valisation/test dataset for two stage spanning tree pipelines with a minimum weight spanning tree CO layer
"""
function build_dataset(;
    nb_scenarios=5:5:20,
    first_max=20:20,
    second_max=10:5:30,
    seeds=1:5,
    grid_sizes=4:6,
    solver=benders_solver,
    parallel=false,
    negative_weights=false,
)
    train_set_params = []
    for gs in grid_sizes
        for ns in nb_scenarios
            for fm in first_max
                for sm in second_max
                    for seed in seeds
                        push!(train_set_params, (gs, ns, fm, sm, seed))
                    end
                end
            end
        end
    end

    if parallel
        return ThreadsX.collect(
            build_solve_and_encode_instance_as_maximum_weight_spanning_tree_single_scenario_layer(;
                grid_size=gs,
                seed=seed,
                nb_scenarios=ns,
                first_max=fm,
                second_max=sm,
                solver=solver,
                negative_weights=negative_weights,
            ) for (gs, ns, fm, sm, seed) in train_set_params
        )
    end

    return [
        build_solve_and_encode_instance_as_maximum_weight_spanning_tree_single_scenario_layer(;
            grid_size=gs,
            seed=seed,
            nb_scenarios=ns,
            first_max=fm,
            second_max=sm,
            solver=solver,
            negative_weights=negative_weights,
        ) for (gs, ns, fm, sm, seed) in train_set_params
    ]
end

# ### Build training set
"""
function build_or_load_spanning_tree_CO_layer_datasets(
    ;
    only_small=false, 
    parallel=false,
    normalized=true     # To get normalized features
)

Returns three dictionnaries: training_datasets, validation_datasets, test_datasets
Each entry of each dictionnary is of the forme dataset_name, dataset
"""
function build_or_load_spanning_tree_CO_layer_datasets(;
    only_small=false, only_lagrangian=false, parallel=false, normalized=true, negative_weights=false, smaller_datasets=false
)
    options_name = "_norm" * string(normalized) * "_neg" * string(negative_weights)
    if smaller_datasets
        options_name *= "_smaller"
    end

    nb_scenarios = 5:5:20
    first_max = 20:20
    second_max = 10:5:30
    train_seeds = 1:5
    val_seeds = 51:55
    test_seeds = 101:105
    small_sizes = 4:6
    large_sizes = 10:10:60

    if smaller_datasets
        nb_scenarios = 10:5:10
        first_max = 20:20
        second_max = 10:10:30
        train_seeds = 1:2
        val_seeds = 51:52
        test_seeds = 101:102
        small_sizes = 4:5
        large_sizes = 10:10:20
    end

    training_datasets = Dict()
    validation_datasets = Dict()
    test_datasets = Dict()
    datasets = []

    if !only_lagrangian
        @info "1/7: small_benders_train_data"
        training_datasets["small_benders_train_data" * options_name] = build_dataset(;
            nb_scenarios=nb_scenarios,
            first_max=first_max,
            second_max=second_max,
            seeds=train_seeds,
            grid_sizes=small_sizes,
            solver=benders_solver,
            parallel=parallel,
            negative_weights=negative_weights
        )
        push!(
            datasets,
            training_datasets["small_benders_train_data" * options_name],
        )

        @info "2/7: small_benders_val_data"
        validation_datasets["small_benders_val_data" * options_name] = build_dataset(;
            nb_scenarios=nb_scenarios,
            first_max=first_max,
            second_max=second_max,
            seeds=val_seeds,
            grid_sizes=small_sizes,
            solver=benders_solver,
            parallel=parallel,
            negative_weights=negative_weights
        )
        push!(
            datasets,
            validation_datasets["small_benders_val_data" * options_name],
        )

        @info "3/7: small_benders_test_data"
        test_datasets["small_benders_test_data" * options_name] = build_dataset(;
            nb_scenarios=nb_scenarios,
            first_max=first_max,
            second_max=second_max,
            seeds=test_seeds,
            grid_sizes=small_sizes,
            solver=benders_solver,
            parallel=parallel,
            negative_weights=negative_weights
        )
        push!(datasets, test_datasets["small_benders_test_data" * options_name])
    end

    @info "4/7: small_lagrangian_test_data"
    test_datasets["small_lagrangian_test_data" * options_name] = build_dataset(;
        nb_scenarios=nb_scenarios,
        first_max=first_max,
        second_max=second_max,
        seeds=test_seeds,
        grid_sizes=small_sizes,
        solver=lagrangian_heuristic_solver,
        parallel=parallel,
        negative_weights=negative_weights,
    )
    push!(datasets, test_datasets["small_lagrangian_test_data" * options_name])

    if !only_small
        @info "5/7: large_lagrangian_train_data"
        training_datasets["large_lagrangian_train_data" * options_name] = build_dataset(;
            nb_scenarios=nb_scenarios,
            first_max=first_max,
            second_max=second_max,
            seeds=train_seeds,
            grid_sizes=large_sizes,
            solver=lagrangian_heuristic_solver,
            parallel=parallel,
            negative_weights=negative_weights,
        )
        push!(datasets, training_datasets["large_lagrangian_train_data" * options_name])

        @info "6/7: large_lagrangian_val_data"
        validation_datasets["large_lagrangian_val_data" * options_name] = build_dataset(;
            nb_scenarios=nb_scenarios,
            first_max=first_max,
            second_max=second_max,
            seeds=val_seeds,
            grid_sizes=large_sizes,
            solver=lagrangian_heuristic_solver,
            parallel=parallel,
            negative_weights=negative_weights,
        )
        push!(datasets, validation_datasets["large_lagrangian_val_data" * options_name])

        @info "7/7: large_lagrangian_test_data"
        test_datasets["large_lagrangian_test_data" * options_name] = build_dataset(;
            nb_scenarios=nb_scenarios,
            first_max=first_max,
            second_max=second_max,
            seeds=test_seeds,
            grid_sizes=large_sizes,
            solver=lagrangian_heuristic_solver,
            parallel=parallel,
            negative_weights=negative_weights,
        )
        push!(datasets, test_datasets["large_lagrangian_test_data" * options_name])
    end

    if normalized
        σ = compute_σ(training_datasets["small_benders_train_data" * options_name])
        if !only_small
            σ = compute_σ(training_datasets["large_lagrangian_train_data" * options_name])
        end

        for dataset in datasets
            reduce_data!(dataset, σ)
        end
    end

    return training_datasets, validation_datasets, test_datasets
end

## Learning FYL models

"""
function train_save_or_load_FYL_model!(
    ;
    nb_samples,         # nb_samples used in FYL perturbation
    train_data_name,    # name of the training set (used for saving or loading)
    train_data,         # dataset of instance obtained with `build_or_load_spanning_tree_CO_layer_datasets()`
    nb_epochs           # nb_epochs in the stochastic gradient descent
)
"""
function train_save_or_load_FYL_model!(; nb_samples, train_data_name, train_data, nb_epochs)
    mkpath("models")
    filename = joinpath(
        "models",
        "fyl_" *
        train_data_name *
        "_samp" *
        string(nb_samples) *
        "_epochs" *
        string(nb_epochs) *
        ".jld2",
    )

    if isfile(filename)
        dict_read = JLD2.load(filename)
        model = Chain(dict_read["model"], X -> dropdims(X; dims=1))
        training_time = dict_read["training_time"]
        losses = dict_read["losses"]
        return model, training_time, losses
    end

    # # initial_model
    nb_features = size(train_data[1][1], 1)
    model = Chain(Dense(nb_features => 1; bias=false), X -> dropdims(X; dims=1))

    # # Pipeline and loss
    perturbed_maximizer = PerturbedAdditive(
        maximum_weight_spanning_tree_single_scenario_linear_maximizer;
        ε=1.0,
        nb_samples=nb_samples,
    )
    loss = FenchelYoungLoss(perturbed_maximizer)

    # Train the model

    opt = ADAM()
    losses = Float64[]
    training_time = @elapsed @showprogress for epoch in 1:nb_epochs
        l = 0.0
        for (x, y, (inst, lb, ub, sol)) in train_data
            grads = gradient(Flux.params(model)) do
                l += loss(model(x), y; inst=inst)
            end
            Flux.update!(opt, Flux.params(model), grads)
        end
        push!(losses, l)
    end

    lineplot(losses; xlabel="Epoch", ylabel="Loss")

    JLD2.save(
        filename,
        Dict("model" => model, "training_time" => training_time, "losses" => losses),
    )

    return model, training_time, losses
end

## Wrapper around a function to count the number of calls

"""
mutable struct CountFunctionCall{F}
    counter::Int
    const f::F
end

Wrapper around a function to print the number of call to the function
"""
mutable struct CountFunctionCall{F}
    counter::Int
    const f::F
end

"""
Wrapper around a function to print the number of call to the function
"""
function CountFunctionCall(f)
    return CountFunctionCall(0, f)
end

function (c::CountFunctionCall)(args...)
    c.counter += 1
    println(c.counter)
    return c.f(args...)
end

## Black box loss

"""
function train_save_load_BBL_model(
    ;
    train_data_name,            # Data name (to save on disk)
    train_data,                 # Data obtained with `build_or_load_spanning_tree_CO_layer_datasets()`
    nb_DIRECT_iterations=1000,  # Nb iterations of the black box algorithm (DIRECT)
    perturbed=false,            # Activate Gaussian perturbation
    nb_perturbations=20,        # Nb pertubation scenarios
    perturbation_intensity=0.1, # Strength of the pertubation
    force_recompute=false       # learn even if model available on disk
)
"""
function train_save_load_BBL_model(;
    train_data_name,
    train_data,
    nb_DIRECT_iterations=1000,
    perturbed=false,
    nb_perturbations=20,
    perturbation_intensity=0.1,
    force_recompute=false,
)
    mkpath("models")
    modelname =
        "bbl_" *
        train_data_name *
        "_iter" *
        string(nb_DIRECT_iterations) *
        "_pert" *
        string(perturbed)
    if perturbed
        modelname *=
            "_intpert" *
            string(perturbation_intensity) *
            "_nbpert" *
            string(nb_perturbations)
    end
    filename = joinpath("models", modelname * ".jld2")

    if isfile(filename) && !force_recompute
        results = JLD2.load(filename)
        u = results["u"]
        training_time = results["training_time"]

    else
        training_instances = [(x, inst) for (x, _, (inst, _, _, _)) in train_data]

        scaling_function(inst::TwoStageSpanningTreeInstance) = float(nv(inst.g) - 1)

        function maximizer(θ::AbstractMatrix, inst::TwoStageSpanningTreeInstance)
            forest, _ = kruskal_maximum_weight_spanning_tree_single_scenario(
                θ, inst
            )
            return forest
        end

        decoder_function = (forest, inst) -> forest

        function cost_function(forest, inst)
            return evaluate_first_stage_solution(
                inst, forest
            )
        end

        loss = BlackBoxLoss(
            training_instances,
            maximizer,
            decoder_function,
            cost_function;
            scaling_function=scaling_function,
            perturbed=perturbed,
            perturbation_intensity=perturbation_intensity,
            nb_perturbations=nb_perturbations,
        )

        # ## Optimize this `BlackBoxLoss` using the `DIRECT` algorithm of `NLopt`.
        u0 = zeros(loss.nb_features)

        println("Black_box_loss_test")
        println(black_box_loss(u0, loss))

        prob = Optimization.OptimizationProblem(
            CountFunctionCall(black_box_loss),
            u0,
            loss;
            lb=-1.0 * ones(loss.nb_features),
            ub=1.0 * ones(loss.nb_features),
        )
        training_time = @elapsed sol = OptimizationNLopt.solve(
            prob, OptimizationNLopt.NLopt.GN_DIRECT_L(); maxiters=nb_DIRECT_iterations
        )

        println(sol)

        # Save
        JLD2.save(filename, Dict("u" => sol.u, "training_time" => training_time))
        u = sol.u
    end

    # Turn result into a Flux model
    model = Chain(Dense((u)', false), x -> dropdims(x; dims=1))

    return modelname, model, training_time
end

"""
function train_bbl_model_and_add_to_dict!(
    models                      # Dictionnary to which the result will be added
    ;
    train_data_name,            # Data name (to save on disk)
    train_data,                 # Data obtained with `build_or_load_spanning_tree_CO_layer_datasets()`
    nb_DIRECT_iterations=1000,  # Nb iterations of the black box algorithm (DIRECT)
    perturbed=false,            # Activate Gaussian perturbation
    nb_perturbations=20,        # Nb pertubation scenarios
    perturbation_intensity=0.1, # Strength of the pertubation
    force_recompute=false       # learn even if model available on disk
)
"""
function train_bbl_model_and_add_to_dict!(
    models;
    train_data_name,
    train_data,
    nb_DIRECT_iterations=1000,
    perturbed=false,
    nb_perturbations=20,
    perturbation_intensity=0.1,
    force_recompute=false,
)
    modelname, model, training_time = train_save_load_BBL_model(;
        train_data_name=train_data_name,
        train_data=train_data,
        nb_DIRECT_iterations=nb_DIRECT_iterations,
        perturbed=perturbed,
        nb_perturbations=nb_perturbations,
        perturbation_intensity=perturbation_intensity,
        force_recompute=force_recompute,
    )
    return models[modelname] = Dict(
        "model" => model,
        "learning_algorithm" => "bbl",
        "train_data_name" => train_data_name,
        "training_time" => training_time,
        "pert" => perturbed,
        "intpert" => perturbation_intensity,
        "nbpert" => nb_perturbations,
    )
end

## Evaluation function

function pipeline_of_model(model)
    return (x, inst) ->
        kruskal_maximum_weight_spanning_tree_single_scenario_first_stage_forest(
            model(x), inst
        )
end

function evaluate_model_gap_on_dataset_first_instance(model, dataset)
    pipeline = pipeline_of_model(model)
    x, y, (inst, lb, ub, sol) = evaluate_first_stage_solution(inst, pipeline(x, inst))
    return (result - lb) / lb
end

"""
function test_models_on_data_set(
    ;
    models,    # Model name
    dataset,   # Dataset obtained with `build_or_load_spanning_tree_CO_layer_datasets()`  
)

returns an array, each instance of which contains a tuple (inst, lb, UBs) where
 - inst is an instance 
 - lb is the lower bound (lagrangian or optimal solution) 
 - Ubs  is a dictionnary with the ub (lagrangian or exaxt), and the performance of every model on the instance
"""
function test_models_on_data_set(;
    models,    # Model name
    dataset,        # Dataset obtained with `build_or_load_spanning_tree_CO_layer_datasets()`  
)
    results = []
    @showprogress for (x, y, (inst, lb, ub, sol)) in dataset
        inst_results = Dict()
        inst_results["ub"] = ub
        no_first_stage_cost = evaluate_first_stage_solution(inst, [])
        for (modelname, model_dict) in models
            model = model_dict["model"]
            pipeline = pipeline_of_model(model)
            pipeline_cost = evaluate_first_stage_solution(
                inst, pipeline(x, inst)
            )
            inst_results[modelname] = min(no_first_stage_cost, pipeline_cost)
        end
        push!(results, (inst, lb, inst_results))
    end
    return results
end

"""
function test_or_load_models_on_datasets(;models, datasets, results_folder,recompute_results=false)

returns a dict with the results of all the models on each dataset. Results are saved. If results file exists, loads it except if `recompute_results=true`
"""
function test_or_load_models_on_datasets(;models, datasets, results_folder,recompute_results=false)
    results = Dict()
    for (name, dataset) in datasets
        result_filename = joinpath(results_folder, name * ".jld2")
        if !recompute_results && isfile(result_filename)
            results[name] = JLD2.load(result_filename)[name]
        else
            results[name] = test_models_on_data_set(; models=models, dataset=dataset)
            JLD2.save(result_filename, Dict(name => results[name]))
        end
    end
    return results
end