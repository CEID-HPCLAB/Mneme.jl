mutable struct OneHotEncoder
    categories::Union{Vector, String}
    drop::Union{String, Vector, Nothing}
    encoder::Py
    file::String
    features::Vector{Symbol}
    feature_idxs::Vector{Int}
    
end

_py_dtype(::Type{Float64}) = np.float64
_py_dtype(::Type{Float32}) = np.float32
_py_dtype(::Type{Int64})   = np.int64
_py_dtype(::Type{Int32})   = np.int32
_py_dtype(::Nothing)       = nothing

OneHotEncoder(
    file::String,
    features::Vector{Symbol};
    categories::Union{Vector, String} = "auto",
    drop::Union{String, Vector{String}, Nothing} = nothing,
    dtype::Union{Type, Nothing} = Float64,
    handle_unknown::String = "error",
    sparse_output::Bool = true
) =
begin
    py_dtype = _py_dtype(dtype)
    OneHotEncoder(
        categories,
        drop,
        sklearn.OneHotEncoder(
            categories = categories isa String ? categories : [np.array(cat_feat) for cat_feat in categories],
            drop = drop,
            dtype = py_dtype,
            handle_unknown = handle_unknown,
            sparse_output = sparse_output
        ),
        file,
        features,
        Int[],
    )
end

function fit(encoder::OneHotEncoder, reader::BlockReader)
    if !isa(encoder.categories, String)
        encoder.encoder.categories_ = encoder.encoder.categories
        encoder.encoder.n_features_in_ = length(encoder.categories)
        encoder.encoder.feature_names_in_ = encoder.features
        
        encoder.encoder._missing_indices = Dict{Int, Int}()
        encoder.encoder._infrequent_enabled = false

        _set_internal_onehot_state!(encoder.encoder)
        return
    end

    offsets = reader.block_offsets
    encoder.feature_idxs = _map_features(encoder.features, reader.feature_idxs_map)

    file = encoder.file; features = encoder.features
    args = (file, encoder.feature_idxs)

    offsets_bounds = [(offsets[i], offsets[i+1]) for i in 1:length(offsets)-1]
    push!(offsets_bounds, (offsets[end], -1))
    
    _partial_res = torcjulia.map(
        _partial_fit_ohe,
        offsets_bounds;
        chunksize = 1,
        args = args
    )

    _set_attributes_ohe(encoder.encoder, _reduce_ohe(_partial_res), features, encoder.feature_idxs)

end

function _set_attributes_ohe(encoder::Py, stats::Tuple{Py, Dict{Int, Int}}, features::Vector{Symbol}, feature_idxs::Vector{Int})
    encoder.categories_ = stats[1]
    encoder.n_features_in_ = length(features)
    encoder.feature_names_in_ = features[sortperm(feature_idxs)]
    
    encoder._missing_indices = stats[2] 
    encoder._infrequent_enabled = false
    
    _set_internal_onehot_state!(encoder)

end

function set_missing_indices(categories::Vector{<:Vector})
    missing_indices = Dict{Int, Int}()

    for (feature_idx, categories_for_idx) in enumerate(categories)
        if any(ismissing.(categories_for_idx))
            missing_indices[feature_idx - 1] = length(categories_for_idx) - 1
        end
    end

    missing_indices

end

function _partial_fit_ohe(offsets::Tuple{Int, Int}, file::String,
                    feat_mapping::Vector{Int})::Vector{Vector{Union{Missing, Any}}}
    X = _fetch_chunk_ohe(offsets, file, feat_mapping)
    
    [unique(col) for col in eachcol(X)]

end

function _fetch_chunk_ohe(offsets::Tuple{Int, Int}, file::String, feat_mapping::Vector{Int})::DataFrame
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

function _reduce_ohe(stats)::Tuple{Py, Dict{Int, Int}}
    cats = stats[1]

    @inbounds for i in 2:length(stats)
        new_cats = stats[i]

        cats = [
            sort(union(new, last))
            for (last, new) in zip(cats, new_cats)
        ]
    end

    missing_indices = set_missing_indices(cats)
    pylist([np.array(cat_feat) for cat_feat in cats]), missing_indices
    
end

function transform(encoder::OneHotEncoder, X)
    warnings = pyimport("warnings")
    warnings.filterwarnings("ignore", message = "X does not have valid feature names")

    # if sparse output is true, Y is always a scipy.sparse._csr.csr_matrix (line 1080 -> /preprocessing/_encoders.py)
    encoder.encoder.transform(np.array(X[:, sort(encoder.feature_idxs)]))

end

function _get_cat_idx(drop_cat::Py, categories::Py)::Int
    pyconvert(Int, np.where(categories .== drop_cat)[0][0])

end

function _set_internal_onehot_state!(encoder::Py)
    n_features = length(encoder.categories_)  
    py_none = pybuiltins.None; drop_param = encoder.drop

    drop_idx = Vector{Union{Int, Nothing}}(undef, n_features)
    fill!(drop_idx, nothing)

    if !pyis(drop_param, py_none)
        drop_param_str = try string(drop_param) catch nothing end

        if drop_param_str == "if_binary"
            drop_idx = [length(cat) == 2 ? 0 : nothing for cat in encoder.categories_]
        elseif drop_param_str == "first"
            drop_idx .= 0
        else
            drop_idx = map(enumerate(drop_param)) do (i, d)
                            d === nothing ? py_none : _get_cat_idx(d, encoder.categories_[i - 1])
                            end
        end
        encoder.drop_idx_ = np.array(drop_idx)
    
    else
        encoder.drop_idx_ = py_none

    end

    encoder._drop_idx_after_grouping = encoder.drop_idx_
    encoder._default_to_infrequent_mappings = pylist([py_none for _ in 1:n_features])
    encoder._infrequent_indices = pydict()

    encoder._n_features_outs = pylist([length(cat) - (drop_idx[i] !== nothing ? 1 : 0) for (i, cat) in enumerate(encoder.categories_)])
    
end

function to_JuliaCSR(X::Py)
    dtype = pyis(np.issubdtype(X.data.dtype, np.floating),  pybuiltins.True) ?
            (pyis(X.data.dtype, np.float32) ? Float32 : Float64) : Int
    data    = pyconvert(Vector{dtype}, X.data)
    indices = pyconvert(Vector{Int}, X.indices) .+ 1  # +1 for 1-based indexing
    indptr  = pyconvert(Vector{Int}, X.indptr) .+ 1   # +1 for 1-based indexing
    m, n   = pyconvert(Tuple{Int, Int}, X.shape)

    sparsecsr(indptr[1:end-1], indices, data, m, n)
    
end

function print_stats(encoder::OneHotEncoder)
    println("categories_: $(encoder.encoder.categories_)")
    println("n_features_in_: $(encoder.encoder.n_features_in_)")
    println("feature_names_in_: $(encoder.encoder.feature_names_in_)")
    println("drop_idx_: $(encoder.encoder.drop_idx_)")

end

_map_features(features::Vector{Symbol}, mapping::Dict{Symbol, Int}) = [mapping[f] for f in features if f in keys(mapping)]

get_encoder(encoder::OneHotEncoder)::Py = encoder.encoder