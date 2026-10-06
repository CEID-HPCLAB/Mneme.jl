using Serialization

struct BlockReader
    filepath::String
    n_rows::Int
    n_cols::Int
    columns::Vector{Symbol}
    feature_idxs_map::Dict{Symbol, Int}
    target::Vector{Symbol}
    block_size::Int
    n_blocks::Int
    block_offsets::Vector{Int}

    function BlockReader(filepath::String; num_blocks::Int = 1, num_rows::Int = -1,
                        target::Vector{Symbol} = Symbol[], offsets_path::String = "")
        
        if !isfile(filepath)
            throw(ArgumentError("CSV file not found: $filepath"))
        end

        if num_rows <= 0 && num_rows != -1
            throw(ArgumentError("num_rows must be positive, got: $num_rows"))
        end

        if num_blocks <= 0
            throw(ArgumentError("num_blocks must be positive, got: $num_blocks"))
        end
        
        n_rows = num_rows == -1 ? inspect_rows(filepath) : num_rows; n_cols, columns, feat_map = inspect_cols(filepath)

        block_size = ceil(Int, n_rows / num_blocks); target = isempty(target) ? [columns[end]] : target

        offsets_path = (!isempty(offsets_path) && !endswith(offsets_path, ".dat")) ? offsets_path * ".dat" : offsets_path

        offsets = (!isempty(offsets_path) && isfile(offsets_path)) ? fetch_block_offsets(offsets_path) : create_block_offsets(filepath, n_rows, block_size)

        new(filepath, n_rows, n_cols, columns, feat_map, target, block_size, length(offsets), offsets)
    end

end

function create_block_offsets(path::String, nrows::Int, block_size::Int)
    offsets = Int[]
    try
        open(path, "r") do file
            readline(file); offset = position(file); row = 0
            for line in eachline(file)
                row += 1
                if (row - 1) % block_size == 0
                    push!(offsets, offset)
                end
                offset += sizeof(line) + 1 
                if row == nrows
                    break
                end
            end
        end
    catch e
        throw(ErrorException("Failed to create block offsets from '$path': $(string(e))"))
    end
    
    offsets
end

function fetch_block_offsets(filename::String)::Vector{UInt64}
    offsets = UInt64[]
    open(filename, "r") do io
        while !eof(io)
            push!(offsets, read(io, UInt64))
        end
    end
    
    offsets
end

function get_offsets(reader::BlockReader)
    reader.block_offsets
end

function get_num_blocks(reader::BlockReader)
    reader.n_blocks
end

function save_offsets(offsets::Vector{Int64}; path::String = "offsets.dat")
    path = endswith(path, ".dat") ? path : path * ".dat"
    path = abspath(path)

    dir = dirname(path)
    isdir(dir) || throw(ArgumentError(
        "Directory '$dir' does not exist."
    ))

    open(path, "w") do io
        for offset in offsets
            write(io, offset)
        end
    end

    nothing
end

function inspect_cols(path::String)

    header = readline(path)
    cols = Symbol.(split(header, ','))

    feature_idxs_map = Dict(col => i for (i, col) in enumerate(cols))

    length(cols), cols, feature_idxs_map
end

function inspect_rows(path::String)
    try
        output = read(`wc -l $path`, String)
        n = parse(Int, split(output)[1])
        return n - 1 
    catch
        cnt = 0
        open(path, "r") do f
            bufsize = 1 << 16  
            buf = Vector{UInt8}(undef, bufsize)
            while !eof(f)
                nread = read!(f, buf)
                count += count(c -> c == UInt8('\n'), buf[1:nread])
            end
        end
        return cnt - 1
    end
end