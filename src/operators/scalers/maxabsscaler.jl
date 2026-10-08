 mutable struct MaxAbsScaler
    scaler::Py
    file::String
    features::Vector{Symbol}
    feature_idxs::Vector{Int}

end

MaxAbsScaler(file::String, features::Vector{Symbol}; copy = true) =
    MaxAbsScaler(
        sklearn.MaxAbsScaler(copy = copy),
        file,
        features,
        Int[],
    )

function fit(scaler::MaxAbsScaler, reader::BlockReader)
    offsets = reader.block_offsets
    scaler.feature_idxs = _map_features(scaler.features, reader.feature_idxs_map)
    
    file = scaler.file; features = scaler.features
    args = (file, scaler.feature_idxs)

    offsets_bounds = [(offsets[i], offsets[i+1]) for i in 1:length(offsets)-1]
    push!(offsets_bounds, (offsets[end], -1))
    
    _partial_res = torcjulia.map(
        _partial_fit_maxabs,
        offsets_bounds;
        chunksize = 1,
        args = args
    )
    
    _set_attributes_maxabs(scaler.scaler, _reduce_maxabs(_partial_res), features, scaler.feature_idxs)

end

function _set_attributes_maxabs(scaler::Py, stats::Tuple{Py, Int}, features::Vector{Symbol}, feature_idxs::Vector{Int})
    scaler.max_abs_ = stats[1]; scaler.n_samples_seen_ = stats[2]
    scaler.scale_ = stats[1]
    scaler.n_features_in_ = length(stats[1])
    scaler.feature_names_in_ = features[sortperm(feature_idxs)]

end

function _partial_fit_maxabs(offsets::Tuple{Int, Int}, file::String,
                      feat_mapping::Vector{Int})::Tuple{Vector{Float64}, Int}
    X = _fetch_chunk_maxabs(offsets, file, feat_mapping)
    
    n_samples = nrow(X)
    max_abs = Base.map(col -> maximum(abs, col), eachcol(X))

    
    max_abs, n_samples

end

function _fetch_chunk_maxabs(offsets::Tuple{Int, Int}, file::String, feat_mapping::Vector{Int})::DataFrame
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

function _reduce_maxabs(stats::AbstractVector{<:Tuple{AbstractVector{<:Number}, Integer}})
    maxabs = stats[1][1]; total_samples = stats[1][2]

    @inbounds for i in 2:length(stats)
        s = stats[i]
        @. maxabs = max(maxabs, s[1])
        total_samples += s[2]
    end

    np.array(maxabs), total_samples

end

function print_stats(scaler::MaxAbsScaler)
    println("scale_: $(scaler.scaler.scale_)\
      \nn_features_in_: $(scaler.scaler.n_features_in_)\
      \nn_samples_seen_: $(scaler.scaler.n_samples_seen_)\
      \nfeature_names_in_: $(scaler.scaler.feature_names_in_)")

end

_map_features(features::Vector{Symbol}, mapping::Dict{Symbol, Int}) = [mapping[f] for f in features if f in keys(mapping)]

get_scaler(scaler::MaxAbsScaler)::Py = scaler.scaler                                                             