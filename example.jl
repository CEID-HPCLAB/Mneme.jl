using torcjulia

include(joinpath(@__DIR__, "src", "mneme.jl"))
import .Mneme

using Printf

# Input Dataset Structure
# 100 samples
# 3 Input Features: (1 categorical, 2 numerical) -> [Name, Age, Score]

const OFFSETS_FILEPATH = "./data/offsets.dat" # path to the offsets file for the dataset
const DATASET_FILEPATH = "./data/data.csv" # path to the dataset CSV file
const NROWS = 100 # total number of dataset's samples
const NCOLS = 3 # total number of dataset's input features
const NUM_BLOCKS = 10 # number of blocks to split the dataset into

function main()
    reader = Mneme.BlockReader(DATASET_FILEPATH; num_blocks = NUM_BLOCKS, 
                               num_rows = NROWS, offsets_path = OFFSETS_FILEPATH)
    
    # Uncomment the following line to save the offsets to a file, if not already saved                           
    # Mneme.save_offsets(reader.block_offsets; path = OFFSETS_FILEPATH) 
    
    std_scaler = Mneme.StandardScaler(DATASET_FILEPATH, reader.columns[2:3])
    Mneme.fit(std_scaler, reader)
    
    Mneme.print_stats(std_scaler)

    minmax_scaler = Mneme.MinMaxScaler(DATASET_FILEPATH, reader.columns[2:3])
    Mneme.fit(minmax_scaler, reader)

    Mneme.print_stats(minmax_scaler)

    ordinal_encoder = Mneme.OrdinalEncoder(DATASET_FILEPATH, [reader.columns[1]])
    Mneme.fit(ordinal_encoder, reader)

    Mneme.print_stats(ordinal_encoder)

    label_encoder = Mneme.LabelEncoder(DATASET_FILEPATH, reader.columns[end])
    Mneme.fit(label_encoder, reader)

    Mneme.print_stats(label_encoder)
    
    pipeline = Mneme.Pipeline([Mneme.MaxAbsScaler(DATASET_FILEPATH, reader.columns[2:3]),
                              Mneme.LabelEncoder(DATASET_FILEPATH, reader.columns[end])], 
                              DATASET_FILEPATH)
    Mneme.fit(pipeline, reader)

    Mneme.print_stats(pipeline)
end

torcjulia.init(main)