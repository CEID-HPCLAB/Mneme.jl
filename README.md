# 🔥 Mneme.jl: Parallel Out-of-Core Tabular Data Preprocessing in Julia
 [![GitHub release](https://img.shields.io/github/v/release/CEID-HPCLAB/Mneme.jl?include_prereleases&color=%238FD9FB)](https://github.com/CEID-HPCLAB/Mneme.jl/releases)
[![License](https://img.shields.io/badge/License-Apache--2.0-FFDEAD)](https://www.apache.org/licenses/LICENSE-2.0)  <br>
[![Julia](https://img.shields.io/badge/Julia-1.12-purple?logo=julia&logoColor=white)](https://julialang.org)
![HPC](https://img.shields.io/badge/HPC-228B22?style=flat&logo=dna&logoColor=white)
![Data Preprocessing](https://img.shields.io/badge/Data%20Preprocessing-001594?style=flat&logo=dna&logoColor=white)
![torcjulia](https://img.shields.io/badge/torcjulia-C94F00?style=flat&logo=dna&logoColor=white)

**Mneme.jl** is a high-level parallel preprocessing framework for large-scale tabular data. Built on top of [torcjulia](https://github.com/CEID-HPCLAB/torcjulia), Mneme.jl employs an MPI-based hybrid parallelism scheme inspired by the MapReduce paradigm, which combines MPI multiprocessing and Julia-native multithreading to efficiently support out-of-core preprocessing of tabular datasets on multi-node systems (clusters).

Mneme.jl can be considered as the Julia implementation of the [Mneme](https://github.com/CEID-HPCLAB/Mneme) Python library. The project is under active development, and some features of the original Python library are not yet supported.  

> [!NOTE]
> For a detailed description of the Mneme Python framework, we refer the reader to the [original paper](https://link.springer.com/article/10.1007/s10766-026-00817-7) published in the *International Journal of Parallel Programming (IJPP)*. An early version of Mneme was also presented at the [18th International Symposium on High-level Parallel Programming and Applications (HLPP 2025)](https://hlpp-conference.github.io/hlpp-2025/), Innsbruck, Austria, 3-4 July 2025.

## Table of Contents
- [Installation](#installation)
- [API](#api)
- [Demo Example](#demo-example)
- [File Structure](#file-structure)
- [Planned Features](#planned-features)
- [Contact](#contact)

## Installation

Begin by cloning the repository to your local machine and navigate to project's root folder:
```bash
git clone https://github.com/CEID-HPCLAB/Mneme.jl.git
cd Mneme.jl
```

`Mneme.jl` is built on top of `torcjulia`. Therefore, make sure that the [torcjulia](https://github.com/CEID-HPCLAB/torcjulia) repository is also cloned before proceeding:
```bash
git clone https://github.com/CEID-HPCLAB/torcjulia.git
```

Now, dependencies of `Mneme.jl` can be installed using the Julia package manager. Enter the Pkg REPL mode by typing "]" in the Julia `REPL` and then run:
```julia
pkg> activate ./torcjulia/torcjulia
pkg> instantiate
pkg> precompile

pkg> activate .
pkg> develop ./torcjulia/torcjulia
pkg> instantiate
```

## API
Built on top of the [scikit-learn](https://scikit-learn.org/stable/) ecosystem, `Mneme.jl` adopts the widely used `fit()-transform()` API model, offering a high-level and user-friendly interface. It provides a wide range of data preprocessing techniques. Specifically, the operators provided by `Mneme.jl` are summarized in the following table.

<br>

| Operator         | Description |
|:-----------------:|:-------------:|
| [StandardScaler](./src/operators/scalers/standardscaler.jl)   | Z-score normalization |
| [MinMaxScaler](./src/operators/scalers/minmaxscaler.jl)     | Scaling in a range specified by min and max values |
| [MaxAbsScaler](./src/operators/scalers/maxabsscaler.jl)      | Scaling in range [-1, 1] |
| [OneHotEncoder](./src/operators/encoders/onehotencoder.jl)     | Binary transformation for nominal categorical features |
| [OrdinalEncoder](./src/operators/encoders/ordinalencoder.jl)    | Encoding of hierarchical categorical features |
| [LabelEncoder](./src/operators/encoders/labelencoder.jl)      | Encoding of categorical target variables |

<br>

Mneme also provides a [pipeline structure](./src/operators/pipeline.jl) which enables the integration of various operators.

## Demo Example
The code listing below illustrates a representative example of applying Z-Score Normalization to 700 features of a dataset using the corresponding operator in `Mneme.jl`.

Initially, a `BlockReader` component is employed to partition the dataset into blocks. The path to a binary block offset file is provided, specifying where the computed offsets will be stored (`block_offset_save`). This file can be reused in subsequent runs, enabling faster preprocessing pipeline execution without recomputing the block offsets. 

Next, a `StandardScaler` is instantiated and fitted to the specified 700 input features (`num_idxs`). The result is a fitted `StandardScaler`, ready to be used in subsequent pipeline stages for *on the fly* data transformation during batch-level fetching. 

By instantiating the corresponding operator object, the same procedure can be used to fit any standalone operator listed in the table above.

```julia
using torcjulia

include(joinpath(@__DIR__, "src", "mneme.jl"))
import .Mneme

# dataset shape: 10M rows, 701 features (x0, x1, ..., x699, y0)
DATAFILE = "/path/to/data.csv"
NUM_BLOCKS = 600

function main()
    reader = Mneme.BlockReader(DATAFILE; num_blocks = NUM_BLOCKS)

    num_idxs = reader.columns[1:end-1]

    standard_scaler = Mneme.StandardScaler(DATAFILE, num_idxs)
    Mneme.fit(standard_scaler, reader)
end

torcjulia.init(main)
```

## File Structure
```
Mneme.jl/
├── data/
│   ├── data.csv          # Toy dataset used by the example
│   └── offsets.dat       # Binary block-offset file for the toy dataset
│
├── src/                  # Core implementation of Mneme.jl
│
└── example.jl            # End-to-end example demonstrating Mneme.jl operators
```

## Planned Features

- [ ] Improve API documentation
- [ ] Automate and optimize block-size selection
- [ ] Integrate GPU acceleration
- [ ] Add new preprocessing operators

## Contact

For questions, bug reports, or contributions, please open an issue or contact:

- Argiris Sofotasios — a.sofotasios@ac.upatras.gr
