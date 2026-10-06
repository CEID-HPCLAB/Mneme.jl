mutable struct StandardScaler
    params::NamedTuple{(:with_mean, :with_std), Tuple{Bool, Bool}}
    scaler::Py
    file::String
    features::Vector{Symbol}
    feature_idxs::Vector{Int}
    
end

StandardScaler(file::String, features::Vector{Symbol}; copy = true, with_mean = true,
            with_std = true) =
    StandardScaler(
        (with_mean = with_mean, with_std = with_std),
        sklearn.StandardScaler(copy = copy, with_mean = with_mean, with_std = with_std),
        file,
        features,
        Int[],
    )

function fit(scaler::StandardScaler, reader::BlockReader)
    (scaler.params.with_mean || scaler.params.with_std) || (scaler.scaler.mean_ = nothing; scaler.scaler.var_ = nothing; scaler.scaler.scale_ = nothing; return) 

    offsets = reader.block_offsets
    scaler.feature_idxs = _map_features(scaler.features, reader.feature_idxs_map)
    
    file = scaler.file; features = scaler.features
    args = (file, scaler.feature_idxs, scaler.params.with_std)
    
    offsets_bounds = [(offsets[i], offsets[i+1]) for i in 1:length(offsets)-1]
    push!(offsets_bounds, (offsets[end], -1))
    
    _partial_res = torcjulia.map(
        _partial_fit_std,                                                                                                                               
        offsets_bounds;
        chunksize = 1,
        args = args
    )
 
    _set_attributes_std(scaler.scaler, _reduce_std(_partial_res), features, scaler.feature_idxs)

end

function _set_attributes_std(scaler::Py, stats::Tuple{Py, Union{Py, Nothing}, Union{Py, Nothing}, Int}, features::Vector{Symbol}, feature_idxs::Vector{Int})
    scaler.mean_ = stats[1]; scaler.var_ = stats[2]; scaler.scale_ = stats[3]; scaler.n_samples_seen_ = stats[4]
    scaler.n_features_in_ = length(stats[1])
    scaler.feature_names_in_ = features[sortperm(feature_idxs)]

    nothing

end

function _partial_fit_std(offsets::Tuple{Int, Int}, file::String, feat_mapping::Vector{Int},
                          with_std::Bool)::Tuple{Vector{Float64}, Union{Vector{Float64}, Nothing}, Int}
    X = _fetch_chunk_std(offsets, file, feat_mapping)

    n_samples = size(X, 1)

    _ncols = length(feat_mapping)

    _mean = Vector{Float64}(undef, _ncols)

    _var = with_std ? Vector{Float64}(undef, _ncols) : nothing

    @inbounds for j in 1:_ncols

        _mean[j] = mean(X[:, j])

        if with_std
            _var[j] = mean((X[:, j] .- _mean[j]).^2)
        end

    end

    _mean, _var, n_samples

end

function _fetch_chunk_std(offsets::Tuple{Int, Int}, file::String, feat_mapping::Vector{Int})::DataFrame
    open(file, "r") do io
        seek(io, offsets[1])
        buf = offsets[2] === -1 ? read(io) : read(io, offsets[2] - offsets[1])

        data = CSV.read(
            IOBuffer(buf),
            DataFrame;
            header = false,
            select = feat_mapping,
            types = Float64, 
            ntasks = 1,
        )
    end

end

function _reduce_std(stats::Vector{<:Tuple{Vector{Float64}, Union{Vector{Float64}, Nothing}, Int}})::Tuple{Py, Union{Py, Nothing}, Union{Py, Nothing}, Int}
    means = stats[1][1]; vars = stats[1][2]; total_samples = stats[1][3]

    @inbounds for i in 2:length(stats)
        s = stats[i]
        means, vars = _incr_mean_var(means, vars, total_samples, s[1], s[2], s[3])
        total_samples += s[3]
    end

    means = np.array(means)
    if !isnothing(vars)
        stds = np.array(sqrt.(vars)); vars = np.array(vars)
    else
        stds = nothing
    end

    means, vars, stds, total_samples

end

function _incr_mean_var(mean_1, var_1, n1, mean_2, var_2, n2)
    upd_samples = n1 + n2
    sum_1 = mean_1 .* n1; sum_2 = mean_2 .* n2

    upd_mean = (sum_1 + sum_2) ./ upd_samples

    if isnothing(var_1)
        upd_var = nothing
    else
        corr_term = (n1 * n2 / upd_samples) .* (mean_1 - mean_2).^2
        S1 = var_1 .* n1; S2 = var_2 .* n2
        upd_var = (S1 + S2 + corr_term) ./ upd_samples
    end

    upd_mean, upd_var

end

function print_stats(scaler::StandardScaler)
    println("mean_: $(scaler.scaler.mean_)\nscale_: $(scaler.scaler.scale_)\nvar_: $(scaler.scaler.var_)\
            \nn_features_in_: $(scaler.scaler.n_features_in_)\nn_samples_seen_: $(scaler.scaler.n_samples_seen_)\
            \nfeature_names_in_: $(scaler.scaler.feature_names_in_)")

end

_map_features(features::Vector{Symbol}, mapping::Dict{Symbol, Int}) = [mapping[f] for f in features if f in keys(mapping)]

get_scaler(scaler::StandardScaler)::Py = scaler.scaler