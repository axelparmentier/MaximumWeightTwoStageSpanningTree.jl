
# # *Learning structured approximations of combinatorial optimization problems* - paper experiments

# ## Load packages

using UnicodePlots
using MaximumWeightTwoStageSpanningTree
using Graphs

# ## Build training and test set

# ### Choose dataset parameters
# This is the only block you should modify
# Set `only_small` to `false` and `smaller_datasets` to `false` to run the experiments in the paper. Beware, it takes several days to run.

only_small = true # if true, only small instances are considered
smaller_datasets = true # if true, fewer instances are considered computation takes roughly one hour, if false several days
normalized = true

# ### Build datasets

@time training_datasets, validation_datasets, test_datasets = build_or_load_spanning_tree_CO_layer_datasets(;
    parallel=false, # Use true only if solutions have already been computed in folder data, otherwise it may bug due to the MILP solver
    only_small=only_small,
    only_lagrangian=false,
    normalized=normalized,
    negative_weights=true,
    smaller_datasets=smaller_datasets,
);

models = Dict()

# ## Train models

# ### Supervised learning with Fenchel Young Losses (FYL)

for (dataset_name, dataset) in training_datasets
    println("training with FYL on ", dataset_name)
    model, training_time, losses = train_save_or_load_FYL_model!(;
        nb_samples=20, train_data=dataset, train_data_name=dataset_name, nb_epochs=200
    )
    model_name = "fyl_" * dataset_name * "_iter200_perttrue_intpert1_nbpert20"
    models[model_name] = Dict(
        "model" => model,
        "learning_algorithm" => "fyl",
        "train_data_name" => dataset_name,
        "training_time" => training_time,
        "pert" => true,
        "intpert" => 1.0,
        "nbpert" => 20,
        "losses" => losses,
    )
    println(lineplot(losses))
end

# ### Learning by experience with global derivative free algorithm

# ##### Non perturbed

for (dataset_name, dataset) in training_datasets
    train_bbl_model_and_add_to_dict!(
        models; train_data_name=dataset_name, train_data=dataset, nb_DIRECT_iterations=1000
    )
end

# ##### Perturbed

intensities = [0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3]

for intensity in intensities
    for (dataset_name, dataset) in training_datasets
        @info "bbl on " * dataset_name * " with intensity " * string(intensity)
        train_bbl_model_and_add_to_dict!(
            models;
            train_data_name=dataset_name,
            train_data=dataset,
            nb_DIRECT_iterations=1000,
            perturbed=true,
            perturbation_intensity=intensity,
            nb_perturbations=20,
        )
    end
end

# ### Add model corresponding to the approximation algorithm and Lagrangian relaxation heuristic

models["approx"] = Dict(
    "model" => MaximumWeightTwoStageSpanningTree.approx_algorithm_model(),
    "learning_algorithm" => "approx",
    "train_data_name" => "--",
    "training_time" => "--",
    "pert" => false,
    "intpert" => 0,
    "nbpert" => 0,
)

models["ub"] = Dict(
    "learning_algorithm" => "UB",
    "train_data_name" => "--",
    "training_time" => "--",
    "pert" => false,
    "intpert" => 0,
    "nbpert" => 0,
)

# ## Evaluate model performances on validation and test sets 

recompute_results = false # put to true to force results recompute (needed if content of validation or test sets changed)
results_folder = "results"
mkpath(results_folder)
val_and_test_sets = merge(validation_datasets, test_datasets)
results = test_or_load_models_on_datasets(
    models=models,
    datasets=val_and_test_sets,
    results_folder=results_folder,
    recompute_results=recompute_results
)

# ## Build hyperparameters choice table
using Printf

# Function to compute averages on datasets

function average_and_worst_on_dataset(dataset_results, f)
    model_names = collect(keys(dataset_results[1][3]))
    averages = Dict(zip(model_names, zeros(length(model_names))))
    worsts = Dict(zip(model_names, -Inf * ones(length(model_names))))
    for (_, lb, ubs) in dataset_results
        for (model_name, ub) in ubs
            f_val = f(lb, ub)
            averages[model_name] += f_val
            worsts[model_name] = max(worsts[model_name], f_val)
        end
    end
    l = length(dataset_results)
    map!(x -> x / l, values(averages))
    return (averages, worsts)
end

# Hyperparameter tunning table

tables_folder = "tables"
mkpath(tables_folder)

begin
    io_hyperparams = open(joinpath(tables_folder, "hyperparameters_neg.tex"), "w")
    for (name, _) in validation_datasets
        println("Gap on ", name)
        averages, worsts = average_and_worst_on_dataset(
            results[name], (lb, ub) -> (ub - lb) / -lb
        )
        for model in sort(collect(keys(averages)))
            train_data_name = models[model]["train_data_name"]
            learning_algorithm = models[model]["learning_algorithm"]
            intpert = models[model]["pert"] ? models[model]["intpert"] : 0.0
            gap_percent = 100 * averages[model]
            @printf(
                "%s & %s & %.1e & %.1f\\%% \\\\\n",
                train_data_name,
                learning_algorithm,
                intpert,
                gap_percent
            )
            @printf(
                io_hyperparams,
                "%s & %s & %.1e & %.1f\\%% \\\\\n",
                train_data_name,
                learning_algorithm,
                intpert,
                gap_percent
            )
        end
    end
    close(io_hyperparams)
end
model_names = collect(keys(results))


# ## Build Figure evaluating the performance of the model
# Following the results of the hyper-parameter tuning model, we use 1e-3

# #### Starts by choosing the datasets and models used for the figure in accordance with initial dataset selection

performance_dataset_name = "large_lagrangian_test_data_normtrue_negtrue"
performance_bbl_model_name = "bbl_large_lagrangian_train_data_normtrue_negtrue_iter1000_perttrue_intpert0.001_nbpert20"
performance_fyl_model_name = "fyl_large_lagrangian_train_data_normtrue_negtrue_iter200_perttrue_intpert1_nbpert20"
x_nb_vertices = [x^2 for x in 10:10:60]
if smaller_datasets
    x_nb_vertices = [x^2 for x in 10:10:20]
    performance_dataset_name = "large_lagrangian_test_data_normtrue_negtrue_smaller"
    performance_bbl_model_name = "bbl_large_lagrangian_train_data_normtrue_negtrue_smaller_iter1000_perttrue_intpert0.001_nbpert20"
    performance_fyl_model_name = "fyl_large_lagrangian_train_data_normtrue_negtrue_smaller_iter200_perttrue_intpert1_nbpert20"
end
if only_small
    if smaller_datasets
        performance_dataset_name = "small_lagrangian_test_data_normtrue_negtrue_smaller"
        performance_bbl_model_name = "bbl_small_benders_train_data_normtrue_negtrue_smaller_iter1000_perttrue_intpert0.001_nbpert20"
        performance_fyl_model_name = "fyl_small_benders_train_data_normtrue_negtrue_smaller_iter200_perttrue_intpert1_nbpert20"
        "ub"
        x_nb_vertices = [x^2 for x in 4:5]
    else
        performance_dataset_name = "small_lagrangian_test_data_normtrue_negtrue"
        performance_bbl_model_name = "bbl_small_benders_train_data_normtrue_negtrue_iter1000_perttrue_intpert0.001_nbpert20"
        performance_fyl_model_name = "fyl_small_benders_train_data_normtrue_negtrue_iter200_perttrue_intpert1_nbpert20"
        "ub"
        x_nb_vertices = [x^2 for x in 4:6]
    end
end


# Performance of the approximation algorithm
average_and_worst_on_dataset(
    results[performance_dataset_name], (lb, ub) -> (ub - lb) / -lb
)[2]["approx"]

function conditional_average_and_worst_for_model(
    dataset_results, statistic, model_name, condition=instance -> true
)
    count = 0
    average = 0.0
    worst = -Inf
    for (instance, lb, ubs) in dataset_results
        if condition(instance)
            stat_val = statistic(lb, ubs[model_name])
            count += 1
            average += stat_val
            worst = max(worst, stat_val)
        end
    end
    average /= count
    return (average, worst)
end

#  Compute plot data

pipelines_average_worst = [
    conditional_average_and_worst_for_model(
        results[performance_dataset_name],
        (lb, ub) -> (ub - lb) / -lb,
        performance_bbl_model_name,
        instance -> nv(instance.g) == nb_vert,
    ) for nb_vert in x_nb_vertices
]
y_pipeline_average = [y for (y, _) in pipelines_average_worst]
y_pipeline_worst = [y for (_, y) in pipelines_average_worst]

fyl_average_worst = [
    conditional_average_and_worst_for_model(
        results[performance_dataset_name],
        (lb, ub) -> (ub - lb) / -lb,
        performance_fyl_model_name,
        instance -> nv(instance.g) == nb_vert,
    ) for nb_vert in x_nb_vertices
]
y_fyl_average = [y for (y, _) in fyl_average_worst]
y_fyl_worst = [y for (_, y) in fyl_average_worst]

approx_average_worst = [
    conditional_average_and_worst_for_model(
        results[performance_dataset_name],
        (lb, ub) -> (ub - lb) / -lb,
        "approx",
        instance -> nv(instance.g) == nb_vert,
    ) for nb_vert in x_nb_vertices
]
y_approx_average = [y for (y, _) in approx_average_worst]
y_approx_worst = [y for (_, y) in approx_average_worst]

lh_average_worst = [
    conditional_average_and_worst_for_model(
        results[performance_dataset_name],
        (lb, ub) -> (ub - lb) / -lb,
        "ub",
        instance -> nv(instance.g) == nb_vert,
    ) for nb_vert in x_nb_vertices
]
y_lh_average = [y for (y, _) in lh_average_worst]
y_lh_worst = [y for (_, y) in lh_average_worst]

# Plot. Result in folder "figures/gap_to_lag_bound.pdf"

using CairoMakie
CairoMakie.activate!()
figure_path = "figures"
mkpath(figure_path)

begin
    fig = Figure()
    ax = Axis(
        fig[1, 1];
        xlabel="|V|",
        ylabel="Gap to Lagrangian Bound (%)",
        limits=(0, 3600, 0, 6.8),
    )
    lines!(
        ax,
        x_nb_vertices,
        100 * y_pipeline_average;
        label="Pipeline (Regret) average gap",
        linestyle=:dashdot,
    )
    lines!(
        ax,
        x_nb_vertices,
        100 * y_pipeline_worst;
        label="Pipeline (Regret) worst gap",
        linestyle=:dashdot,
    )
    lines!(
        ax,
        x_nb_vertices,
        100 * y_fyl_average;
        label="Pipeline (FYL) average gap",
        linestyle=:dot,
    )
    lines!(
        ax,
        x_nb_vertices,
        100 * y_fyl_worst;
        label="Pipeline (FYL) worst gap",
        linestyle=:dot,
    )
    lines!(ax, x_nb_vertices, 100 * y_lh_average; label="Lagrangian heuristic average gap")
    lines!(ax, x_nb_vertices, 100 * y_lh_worst; label="Lagrangian heuristic worst gap")
    lines!(ax, x_nb_vertices, 100 * y_approx_average; label="Approx algorithm average gap")
    lines!(ax, x_nb_vertices, 100 * y_approx_worst; label="Approx algorithm worst gap")
    axislegend(ax)
    current_figure()
    save(joinpath(figure_path, "gap_to_lag_bound.pdf"), current_figure())
end
