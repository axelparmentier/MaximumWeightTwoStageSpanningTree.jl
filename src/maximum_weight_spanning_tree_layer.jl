function first_stage_cost(
    inst::TwoStageSpanningTreeInstance, e::AbstractEdge, scenario::Int
)
    return inst.first_stage_weights_matrix[src(e), dst(e)]
end

function scenario_second_stage_cost(
    inst::TwoStageSpanningTreeInstance, e::AbstractEdge, scenario::Int
)
    return inst.second_stage_weights[scenario][src(e), dst(e)]
end

function scenario_best_stage_cost(
    inst::TwoStageSpanningTreeInstance, e::AbstractEdge, scenario::Int
)
    return min(
        inst.second_stage_weights[scenario][src(e), dst(e)],
        inst.first_stage_weights_matrix[src(e), dst(e)],
    )
end

function edge_neighbors(g::AbstractGraph, e::AbstractEdge)
    result = Vector{AbstractEdge}()
    for v in [src(e), dst(e)]
        for u in neighbors(g, v)
            push!(result, Edge(min(u, v), max(u, v)))
        end
    end
    return result
end

function compute_minimum_spanning_tree(
    inst::TwoStageSpanningTreeInstance, scenario::Int, weight_function
)
    weights_vec = zeros(ne(inst.g))
    for e in edges(inst.g)
        weights_vec[edge_index(inst, e)] = weight_function(inst, e, scenario)
    end
    weights = get_weight_matrix_from_weight_vector(inst.g, inst.edge_index, weights_vec)
    return kruskal_mst(inst.g, weights)
end

function compute_first_stage_mst(inst::TwoStageSpanningTreeInstance)
    tree = compute_minimum_spanning_tree(inst, -1, first_stage_cost)
    fs_mst_indicator = zeros(ne(inst.g))
    for e in tree
        fs_mst_indicator[edge_index(inst, e)] = 1.0
    end
    return fs_mst_indicator
end

function compute_second_stage_mst(inst::TwoStageSpanningTreeInstance)
    ss_mst_indicator = zeros(ne(inst.g), inst.nb_scenarios)
    for scenario in 1:(inst.nb_scenarios)
        tree = compute_minimum_spanning_tree(inst, scenario, scenario_second_stage_cost)
        for e in tree
            ss_mst_indicator[edge_index(inst, e), scenario] = 1.0
        end
    end
    return ss_mst_indicator
end

function compute_best_stage_mst(inst::TwoStageSpanningTreeInstance)
    bfs_mst_indicator = zeros(ne(inst.g), inst.nb_scenarios)
    bss_mst_indicator = zeros(ne(inst.g), inst.nb_scenarios)
    for scenario in 1:(inst.nb_scenarios)
        tree = compute_minimum_spanning_tree(inst, scenario, scenario_best_stage_cost)
        for e in tree
            e_ind = edge_index(inst, e)
            if inst.first_stage_weights_matrix[src(e), dst(e)] <
                inst.second_stage_weights[scenario][src(e), dst(e)]
                bfs_mst_indicator[e_ind, scenario] = 1.0
            else
                bss_mst_indicator[e_ind, scenario] = 1.0
            end
        end
    end
    return bfs_mst_indicator, bss_mst_indicator
end

function pivot_instance_second_stage_costs(inst::TwoStageSpanningTreeInstance)
    edgeWeights = zeros(ne(inst.g), inst.nb_scenarios)
    for s in 1:(inst.nb_scenarios)
        for e in edges(inst.g)
            edgeWeights[edge_index(inst, e), s] = inst.second_stage_weights[s][
                src(e), dst(e)
            ]
        end
    end
    return edgeWeights
end

function compute_edge_neighbors(inst::TwoStageSpanningTreeInstance)
    neighbors = Vector{Vector{Int}}(undef, ne(inst.g))
    for e in edges(inst.g)
        e_ind = edge_index(inst, e)
        neighbors_list = edge_neighbors(inst.g, e)
        neighbors[e_ind] = Vector{Int}(undef, length(neighbors_list))
        count = 0
        for f in neighbors_list
            count += 1
            neighbors[e_ind][count] = edge_index(inst, f)
        end
    end
    return neighbors
end

"""
    function kruskal_maximum_weight_forest(θ::AbstractVector;inst=inst) 

Returns `(first_stage_forest, second_stage_forest)`, two vectors of edges containing the first stage and second stage edges in a solution of the maximum weeight two stage spanning tree problem with a single second stage scenario given by
"""

function kruskal_maximum_weight_spanning_tree_single_scenario(
    edge_weights::AbstractMatrix, inst::TwoStageSpanningTreeInstance
)
    @assert size(edge_weights) == (ne(inst.g), 2)
    weights = deepcopy(inst.first_stage_weights_matrix)

    for e in edges(inst.g)
        e_ind = edge_index(inst, e)
        w = max(edge_weights[e_ind, 1], edge_weights[e_ind, 2])
        weights[src(e), dst(e)] = w
        weights[dst(e), src(e)] = w
    end

    mst = kruskal_mst(inst.g, weights; minimize=false)
    first_stage_forest = [
        e for e in mst if
        edge_weights[edge_index(inst, e), 1] > edge_weights[edge_index(inst, e), 2]
    ]
    second_stage_forest = [
        e for e in mst if
        edge_weights[edge_index(inst, e), 1] <= edge_weights[edge_index(inst, e), 2]
    ]

    return first_stage_forest, second_stage_forest
end

function kruskal_maximum_weight_spanning_tree_single_scenario_first_stage_forest(
    edge_weights::AbstractMatrix, inst::TwoStageSpanningTreeInstance
)
    return kruskal_maximum_weight_spanning_tree_single_scenario(edge_weights, inst)[1]
end

"""
    function maximum_weight_spanning_tree_single_scenario_linear_maximizer(θ::AbstractMatrix;inst=inst) 

Wrapper around kruskal_maximum_weight_spanning_tree_single_scenario(edge_weights_vector::AbstractVector, inst::TwoStageSpanningTreeInstance) that returns the solution encoded as a vector.
"""
function maximum_weight_spanning_tree_single_scenario_linear_maximizer(
    θ::AbstractMatrix; inst=inst
)
    first_stage_forest, second_stage_forest = kruskal_maximum_weight_spanning_tree_single_scenario(
        θ, inst
    )
    y = zeros(ne(inst.g), 2)
    for e in first_stage_forest
        y[edge_index(inst, e), 1] = 1.0
    end
    for e in second_stage_forest
        y[edge_index(inst, e), 2] = 1.0
    end
    return y
end

"""
    function maximum_weight_spanning_tree_single_scenario_layer_linear_encoder(inst::TwoStageSpanningTreeInstance)
        (inst::TwoStageSpanningTreeInstance)

    returns X::Array{Float64} with `X[f,edge_index(inst,e),s]` containing the value of feature number `f` for edge `e` and scenario `s`

    Features used: (all are homogeneous to a cost)
    - first_stage_cost 
    - second_stage_cost_quantile
    - neighbors_first_stage_cost_quantile 
    - neighbors_scenario_second_stage_cost_quantile 
    - is_in_first_stage_x_first_stage_cost 
    - is_in_second_stage_x_second_stage_cost_quantile 
    - is_first_in_best_stage_x_best_stage_cost_quantile 
    - is_second_in_best_stage_x_best_stage_cost_quantile 

    For features with quantiles, the following quantiles are used: 0:0.1:1
"""
function maximum_weight_spanning_tree_single_scenario_layer_linear_encoder(
    inst::TwoStageSpanningTreeInstance
)

    # Choose quantiles
    quantiles_used = [i for i in 0.0:0.1:1.0]

    # MST features
    fs_mst_indicator = compute_first_stage_mst(inst)
    ss_mst_indicator = compute_second_stage_mst(inst)
    bfs_mst_indicator, bss_mst_indicator = compute_best_stage_mst(inst)
    second_stage_edge_costs = pivot_instance_second_stage_costs(inst)
    edge_neighbors = compute_edge_neighbors(inst)

    # Build features
    nb_features = 3 + 6 * length(quantiles_used)
    X = zeros(Float64, nb_features, ne(inst.g), 2)

    for e in edges(inst.g)
        count_feat = 0
        e_ind = edge_index(inst, e)

        function add_quantile_features(realizations, stage)
            sort!(realizations)
            for p in quantiles_used
                count_feat += 1
                X[count_feat, e_ind, stage] = quantile(realizations, p; sorted=true)
            end
        end

        ## Costs features
        # first_stage_cost
        count_feat += 1
        X[count_feat, e_ind, 1] = inst.first_stage_weights_vector[e_ind]

        # second_stage_mean
        edge_second_stage_costs = [
            inst.second_stage_weights[s][src(e), dst(e)] for s in 1:(inst.nb_scenarios)
        ]

        count_feat += 1
        X[count_feat, e_ind, 2] = mean(edge_second_stage_costs)

        # second_stage_cost_quantile
        add_quantile_features(edge_second_stage_costs, 2)

        ## Neighbors features
        # neighbors_first_stage_cost_quantile,
        edge_neighbors_first_stage_cost_quantile = [
            inst.first_stage_weights_vector[e_i] for e_i in edge_neighbors[e_ind]
        ]
        add_quantile_features(edge_neighbors_first_stage_cost_quantile, 1)

        # neighbors_scenario_second_stage_cost_quantile
        edge_neighbors_scenario_second_stage_cost_quantile = [
            second_stage_edge_costs[n, s] for s in 1:(inst.nb_scenarios) for
            n in edge_neighbors[e_ind]
        ]
        add_quantile_features(edge_neighbors_scenario_second_stage_cost_quantile, 2)

        ## MST features
        # is_in_first_stage_x_first_stage_cost
        count_feat += 1
        X[count_feat, e_ind, 1] =
            fs_mst_indicator[e_ind] * inst.first_stage_weights_vector[e_ind]

        # is_in_second_stage_x_second_stage_cost_quantile
        edge_is_in_second_stage_x_second_stage_cost_quantile = [
            ss_mst_indicator[e_ind, s] * second_stage_edge_costs[e_ind, s] for
            s in 1:(inst.nb_scenarios)
        ]
        add_quantile_features(edge_is_in_second_stage_x_second_stage_cost_quantile, 2)

        # is_first_in_best_stage_x_best_stage_cost_quantile
        edge_is_first_in_best_stage_x_best_stage_cost_quantile = [
            bfs_mst_indicator[e_ind, s] * inst.first_stage_weights_vector[e_ind] for
            s in 1:(inst.nb_scenarios)
        ]
        add_quantile_features(edge_is_first_in_best_stage_x_best_stage_cost_quantile, 1)

        # is_second_in_best_stage_x_best_stage_cost_quantile
        edge_is_second_in_best_stage_x_best_stage_cost_quantile = [
            bss_mst_indicator[e_ind, s] * second_stage_edge_costs[e_ind, s] for
            s in 1:(inst.nb_scenarios)
        ]
        add_quantile_features(edge_is_second_in_best_stage_x_best_stage_cost_quantile, 2)
    end
    return X
end

function approximation_maximum_weight_spanning_tree_single_scenario_layer_linear_w()
    result = zeros(69)
    result[1] = -1.0
    result[2] = -1.0
    return result
end

function approx_algorithm_model()
    approx_weights = approximation_maximum_weight_spanning_tree_single_scenario_layer_linear_w()
    return Chain(Dense((approx_weights)', false), x -> dropdims(x; dims=1))
end

"""
    function build_solve_and_encode_instance_as_maximum_weight_spanning_tree_single_scenario_layer(;
        grid_size=3,
        seed=0,
        nb_scenarios=10, 
        first_max=10, 
        second_max=20, 
        solver=lagrangian_heuristic_solver, 
        load_and_save=true
    )

Builds a two stage spanning tree instance with a square grid graph of width `grid_size`, `seed` for the random number generator, `nb_scenarios` for the second stage, `first_max` and `second_max` as first and second stage maximum weight.

Solves it with `solver`

Encodes it for a pipeline with a maxium weight spanning tree for a two stage instance with single scenario second stage layer.
"""
function build_solve_and_encode_instance_as_maximum_weight_spanning_tree_single_scenario_layer(;
    grid_size=3,
    seed=0,
    nb_scenarios=10,
    first_max=10,
    second_max=20,
    solver=lagrangian_heuristic_solver,
    load_and_save=true,
    negative_weights=false,
)
    inst, lb, ub, sol = build_load_or_solve(;
        grid_size=grid_size,
        seed=seed,
        nb_scenarios=nb_scenarios,
        first_max=first_max,
        second_max=second_max,
        solver=solver,
        load_and_save=load_and_save,
        negative_weights=negative_weights,
    )

    x = maximum_weight_spanning_tree_single_scenario_layer_linear_encoder(inst)

    value, second_stages_sol = evaluate_two_stage_spanning_tree_first_stage_solution_and_compute_second_stage(
        inst, sol
    )
    @assert(abs(ub - value) < 0.00001)
    @assert(length(second_stages_sol) == inst.nb_scenarios)

    y = zeros(ne(inst.g), 2)
    for e in sol
        y[edge_index(inst, e), 1] = 1.0
    end
    for ssol in second_stages_sol
        for e in ssol
            y[edge_index(inst, e), 2] += 1.0 / inst.nb_scenarios
        end
    end

    return x, y, (inst=inst, lb=lb, ub=ub, sol=sol)
end
