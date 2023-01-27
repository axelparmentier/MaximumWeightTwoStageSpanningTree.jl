using MaximumWeightTwoStageSpanningTree
using Test
using Graphs
tol = 0.00001  

@testset "TwoStageSpanningTree.jl" begin
    @testset "Test Cut Generation, Benders and Lagrangian relaxation on a TwoStageSpanningTree instance with a single scenario" begin
        instances = [
            MaximumWeightTwoStageSpanningTree.TwoStageSpanningTreeInstance(
                grid([3, 3]); nb_scenarios=1, first_max=10, second_max=10
            ),
            MaximumWeightTwoStageSpanningTree.TwoStageSpanningTreeInstance(
                grid([3, 3]); nb_scenarios=1, first_max=10, second_max=10, negative_weights=true
            ),
        ]
        for instance in instances
            kruskal_value, _, _ = MaximumWeightTwoStageSpanningTree.kruskal_on_first_scenario_instance(instance)
            benders_value, forest, theta = benders(instance; silent=true)
            @test abs(kruskal_value - benders_value) < tol
    
            cl_val, cl_theta = column_generation(instance)
    
            lb, ub, _, _ = lagrangian_relaxation(instance; nb_epochs=20000)
            @test lb <= ub
            @test lb <= kruskal_value
            @test ub >= kruskal_value
            @test abs(lb - kruskal_value) <= 0.1
            @test abs(
                MaximumWeightTwoStageSpanningTree.lagrangian_dual(-instance.nb_scenarios * cl_theta; inst=instance) - cl_val
            ) <= 0.00001
    
            cut_value, forest = cut_generation(instance; silent=true)
            @test abs(kruskal_value - cut_value) < 0.0001
        end
    end
    
    @testset "Test Cut Generation, Benders and Lagrangian relaxation on a TwoStageSpanningTree instance with 3 scenarios" begin
        instances = [
            MaximumWeightTwoStageSpanningTree.TwoStageSpanningTreeInstance(
                grid([3, 3]); nb_scenarios=3, first_max=10, second_max=10
            ),
            MaximumWeightTwoStageSpanningTree.TwoStageSpanningTreeInstance(
                grid([3, 3]); nb_scenarios=3, first_max=10, second_max=10, negative_weights=true
            ),
        ]
        for instance in instances
            benders_value, forest, theta = benders(instance; silent=true)
            cl_val, cl_theta = column_generation(instance)
            lb, ub, _, _ = lagrangian_relaxation(instance; nb_epochs=20000)
            cut_value, forest = cut_generation(instance; silent=true)
    
            @test abs(cut_value - benders_value) < tol
            @test lb <= ub
            @test lb <= cl_val
            @test cl_val <= benders_value + tol
            @test ub >= benders_value
            @test abs(
                MaximumWeightTwoStageSpanningTree.lagrangian_dual(-instance.nb_scenarios * cl_theta; inst=instance) - cl_val
            ) <= 0.00001
        end
    end
end
