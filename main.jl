# Muhammad Aqmal Khafidz Pratama - 1313621005
# Farhan Maulana Azis - 1313621033

using Serialization, Statistics, LinearAlgebra, MLJ

function get_avg_perclass(dataset)
    class = unique(dataset[:, end])
    class_index = size(dataset, 2)
    feature_size = class_index - 1
    class_size = length(class)
    
    mu_vec = zeros(Float16, 1, feature_size, class_size)
    
    for i = 1:class_size
        c = class[i]
        current_class_pos = (dataset[:, class_index] .- c) .< Float16(0.1)
        current_df = Float32.(dataset[current_class_pos, 1:class_index-1])
        mu = mean(current_df, dims=1)
        mu_vec[1, :, i] = mu
    end
    
    return mu_vec
end

function get_d1_distance(X, mu)
    numclass = size(mu, 3)
    X = repeat(X, outer=[1, 1, numclass])
    subtracted_vector = abs.(X .- mu)
    return subtracted_vector
end

function classify_by_distance_features(X, mu)
    num_instance = size(X, 1)
    mu_vec = repeat(mu, outer=[num_instance, 1, 1])
    dist_vec = get_d1_distance(X, mu_vec)
    min_vector = argmin(dist_vec, dims=3)
    min_index = @. get_min_index(min_vector)
    return min_index
end

function get_min_index(X)
    return X[3]
end

function cascade_classify(dataset, mu)
    class_index = size(dataset, 2)
    feature_size = size(dataset, 2) - 1
    preds = zeros(Int, size(dataset, 1), feature_size)

    for i = 1:feature_size
        current_feature = dataset[:, i]
        current_feature = reshape(current_feature, (size(current_feature, 1), 1))
        current_mu = reshape(mu[1, i, :], (1, 1, size(mu, 3)))
        current_pred = classify_by_distance_features(current_feature, current_mu)

        if i == 1
            preds = current_pred
        else
            preds = hcat(preds, current_pred)
        end
    end

    truth = dataset[:, class_index]
    return truth, preds
end


function confusion_matrix(truth, preds)
    class = unique(truth)
    class_size = length(class)
    valuation = [sum((truth .== class[i]) .& (preds .== class[j])) for i = 1:class_size, j = 1:class_size]
    return valuation
end

function get_true_correctness(valuation)
    return sum(diag(valuation)) / sum(valuation)
end

function classify_and_calculate_correctness(dataset, mu_vector)
    truth, preds = cascade_classify(dataset, mu_vector)
    best_preds = find_best_prediction(preds, truth)
    valuation = confusion_matrix(truth, best_preds)
    correctness = get_true_correctness(valuation)
    return correctness
end

function find_best_prediction(preds, truth)
    best_preds = preds[:, end]

    for i = 1:size(preds, 1)
        matching_index = findfirst(==(truth[i]), preds[i, :])
        if matching_index !== nothing
            best_preds[i] = preds[i, matching_index]
        end
    end

    return best_preds
end

dataset = deserialize("data_9m.mat")

correctness = classify_and_calculate_correctness(dataset, get_avg_perclass(dataset))
println(round(correctness * 100), "%")
