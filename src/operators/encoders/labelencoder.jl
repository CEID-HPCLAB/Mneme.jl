mutable struct LabelEncoder
    encoder::Py
    file::String
    features::Symbol
    feature_idxs::Vector{Int}

end

LabelEncoder(
    file::String,
    features::Symbol
) =
    begin
        LabelEncoder(
            sklearn.LabelEncoder(),
            file,
            features,
            Int[],
        )
    end

function fit(encoder::LabelEncoder, reader::BlockReader)

    offsets= reader.block_offsets
    encoder.feature_idxs = _map_features([encoder.features], reader.feature_idxs_map)
    
    file = encoder.file
    args = (file, encoder.feature_idxs)

    offsets_bounds = [(offsets[i], offsets[i+1]) for i in 1:length(offsets)-1]
    push!(offsets_bounds, (offsets[end], -1))
    
    _partial_res = torcjulia.map(
        _partial_fit_le,
        offsets_bounds;
        chunksize = 1,
        args = args
    )
    
    _set_attributes_le(encoder.encoder, _reduce_le(_partial_res))

end

function _set_attributes_le(encoder::Py, classes::Py)
    encoder.classes_ = classes

end

function _partial_fit_le(offsets::Tuple{Int, Int}, file::String, 
                      feat_mapping::Vector{Int})::Vector{Union{Missing, Any}}
    X = _fetch_chunk_le(offsets, file, feat_mapping)
    
    unique(X[:, 1])

end

function _fetch_chunk_le(offsets::Tuple{Int, Int}, file::String, feat_mapping::Vector{Int})::DataFrame
    open(file, "r") do io
        seek(io, offsets[1])
        buf = offsets[2] === -1 ? read(io) : read(io, offsets[2] - offsets[1])

        data = CSV.read(
            IOBuffer(buf),
            DataFrame;
            header = false,
            select = feat_mapping,
            ntasks = 1,
        )
    end

end

function _reduce_le(stats)::Py
    classes = stats[1]

    for i in 2:length(stats)
        classes = sort!(unique(vcat(classes, stats[i])))
    end

    np.array(classes)

end

function transform(encoder::LabelEncoder, X) 
    warnings = pyimport("warnings")
    warnings.filterwarnings("ignore", message = "X does not have valid feature names")

    encoder.encoder.transform(np.asarray(X[:, encoder.feature_idxs]).ravel())

end

function print_stats(encoder::LabelEncoder)
    println("classes_: $(encoder.encoder.classes_)")
    
end

_map_features(features::Vector{Symbol}, mapping::Dict{Symbol, Int}) = [mapping[f] for f in features if f in keys(mapping)]

get_encoder(encoder::LabelEncoder)::Py = encoder.encoder
