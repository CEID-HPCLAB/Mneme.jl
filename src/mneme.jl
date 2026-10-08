module Mneme

using torcjulia

using DataFrames, CSV, Statistics

using CondaPkg; CondaPkg.add("scikit-learn")
using PythonCall

sklearn = pyimport("sklearn.preprocessing")
np = pyimport("numpy")

include("offsets.jl")

include("./operators/scalers/minmaxscaler.jl")
include("./operators/scalers/maxabsscaler.jl")
include("./operators/scalers/standardscaler.jl")

include("./operators/encoders/ordinalencoder.jl")
include("./operators/encoders/labelencoder.jl")
include("./operators/encoders/onehotencoder.jl")

include("./operators/pipeline.jl")

export BlockReader
export Pipeline
export MinMaxScaler, MaxAbsScaler, StandardScaler
export OrdinalEncoder, OneHotEncoder, LabelEncoder
export fit, print_stats, save_offsets

end