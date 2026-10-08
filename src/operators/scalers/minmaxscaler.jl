mutable struct MinMaxScaler
    scaler::Py
    file::String
    features::Vector{Symbol}
    feature_idxs::Vector{Int}

end

MinMaxScaler(file::String, features::Vector{Symbol}; feature_range = (0.0, 1.0), copy = true, 
            clip = false) =
    MinMaxScaler(
        sklearn.MinMaxScaler(feature_range = feature_range, copy = copy, clip = clip),
        file,
        features,
        Int[],
    )

function fit(scaler::MinMaxScaler, reader::BlockReader)
    offsets = reader.block_offsets
    scaler.feature_idxs = _map_features(scaler.features, reader.feature_idxs_map)
    
    file = scaler.file; features = scaler.features
    args = (file, scaler.feature_idxs)

    offsets_bounds = [(offsets[i], offsets[i+1]) for i in 1:length(offsets)-1]
    push!(offsets_bounds, (offsets[end], -1))
    
    _partial_res = torcjulia.map(
        _partial_fit_mm,
        offsets_bounds;
        chunksize = 1,
        args = args
    )

    _set_attributes_mm(scaler.scaler, _reduce_mm(_partial_res), features, scaler.feature_idxs)

end

function _set_attributes_mm(scaler::Py, stats::Tuple{Py, Py, Int}, features::Vector{Symbol}, feature_idxs::Vector{Int})
    scaler.data_min_ = stats[1]; scaler.data_max_ = stats[2]; scaler.n_samples_seen_ = stats[3]

    range = scaler.feature_range; data_range = stats[2] .- stats[1]; scale = (range[1] - range[0]) ./ data_range
    
    scaler.data_range_ = data_range
    scaler.scale_ = scale
    scaler.min_ = range[0] .- stats[1] .* scale
    scaler.n_features_in_ = length(stats[1])
    scaler.feature_names_in_ = features[sortperm(feature_idxs)]

end

function _partial_fit_mm(offsets::Tuple{Int, Int}, file::String,
                      feat_mapping::Vector{Int})::Tuple{Vector{Float64}, Vector{Float64}, Int}
    X = _fetch_chunk_mm(offsets, file, feat_mapping)
    
    n_samples = nrow(X)
    data_min = Base.map(col -> minimum(col), eachcol(X))
    data_max = Base.map(col -> maximum(col), eachcol(X)) 

    data_min, data_max, n_samples

end

function _fetch_chunk_mm(offsets::Tuple{Int, Int}, file::String, feat_mapping::Vector{Int})::DataFrame
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

function _reduce_mm(stats::AbstractVector{<:Tuple{AbstractVector{<:Number}, AbstractVector{<:Number}, Int}})::Tuple{Py, Py, Int}
    mins = stats[1][1]; maxs = stats[1][2]; total_samples = stats[1][3]

    @inbounds for i in 2:length(stats)
        s = stats[i]
        @. mins = min(mins, s[1])
        @. maxs = max(maxs, s[2])
        total_samples += s[3]
    end
    
    np.array(mins), np.array(maxs), total_samples
    
end

function print_stats(scaler::MinMaxScaler)
    println("min_: $(scaler.scaler.min_)\nscale_: $(scaler.scaler.scale_)\
            \ndata_min_: $(scaler.scaler.data_min_)\ndata_max_: $(scaler.scaler.data_max_)\ndata_range_: $(scaler.scaler.data_range_)\
            \nn_features_in_: $(scaler.scaler.n_features_in_)\nn_samples_seen_: $(scaler.scaler.n_samples_seen_)\
            \nfeature_names_in_: $(scaler.scaler.feature_names_in_)")

end

_map_features(features::Vector{Symbol}, mapping::Dict{Symbol, Int}) = [mapping[f] for f in features if f in keys(mapping)]

get_scaler(scaler::MinMaxScaler)::Py = scaler.scaler
                                                                 