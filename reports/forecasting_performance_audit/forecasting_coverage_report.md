# Forecasting Coverage Audit

## Global status distribution

| status      | count |
| ----------- | ----- |
| missing     | 2161  |
| unavailable | 352   |
| ok          | 259   |

## Status by dataset

| dataset        | status      | count |
| -------------- | ----------- | ----- |
| ganymede       | missing     | 21    |
| ganymede       | ok          | 203   |
| inner_mongolia | missing     | 812   |
| inner_mongolia | ok          | 20    |
| inner_mongolia | unavailable | 8     |
| spe_berg       | missing     | 1133  |
| spe_berg       | ok          | 35    |
| spe_berg       | unavailable | 344   |
| volve          | missing     | 195   |
| volve          | ok          | 1     |

## Unavailable reasons (sample)

    | model   | dataset        | horizon | mode       | well    | reason                                                                                                                   |
| ------- | -------------- | ------- | ---------- | ------- | ------------------------------------------------------------------------------------------------------------------------ |
| timesfm | inner_mongolia | 7       | multi_well |         | TimesFM is not installed. Requires Python <3.12 and JAX. Install via: pip install timesfm (in a Python 3.11 environment) |
| tirex   | inner_mongolia | 7       | multi_well |         | TiRex is not installed. Requires GPU with CUDA ≥8.0. Install via: pip install git+https://github.com/NX-AI/tirex         |
| timesfm | inner_mongolia | 14      | multi_well |         | TimesFM is not installed. Requires Python <3.12 and JAX. Install via: pip install timesfm (in a Python 3.11 environment) |
| tirex   | inner_mongolia | 14      | multi_well |         | TiRex is not installed. Requires GPU with CUDA ≥8.0. Install via: pip install git+https://github.com/NX-AI/tirex         |
| timesfm | inner_mongolia | 30      | multi_well |         | TimesFM is not installed. Requires Python <3.12 and JAX. Install via: pip install timesfm (in a Python 3.11 environment) |
| tirex   | inner_mongolia | 30      | multi_well |         | TiRex is not installed. Requires GPU with CUDA ≥8.0. Install via: pip install git+https://github.com/NX-AI/tirex         |
| timesfm | inner_mongolia | 90      | multi_well |         | TimesFM is not installed. Requires Python <3.12 and JAX. Install via: pip install timesfm (in a Python 3.11 environment) |
| tirex   | inner_mongolia | 90      | multi_well |         | TiRex is not installed. Requires GPU with CUDA ≥8.0. Install via: pip install git+https://github.com/NX-AI/tirex         |
| timesfm | spe_berg       | 7       | multi_well |         | TimesFM is not installed. Requires Python <3.12 and JAX. Install via: pip install timesfm (in a Python 3.11 environment) |
| tirex   | spe_berg       | 7       | multi_well |         | TiRex is not installed. Requires GPU with CUDA ≥8.0. Install via: pip install git+https://github.com/NX-AI/tirex         |
| timesfm | spe_berg       | 7       | per_well   | well_12 | TimesFM is not installed. Requires Python <3.12 and JAX. Install via: pip install timesfm (in a Python 3.11 environment) |
| tirex   | spe_berg       | 7       | per_well   | well_12 | TiRex is not installed. Requires GPU with CUDA ≥8.0. Install via: pip install git+https://github.com/NX-AI/tirex         |
| timesfm | spe_berg       | 7       | per_well   | well_13 | TimesFM is not installed. Requires Python <3.12 and JAX. Install via: pip install timesfm (in a Python 3.11 environment) |
| tirex   | spe_berg       | 7       | per_well   | well_13 | TiRex is not installed. Requires GPU with CUDA ≥8.0. Install via: pip install git+https://github.com/NX-AI/tirex         |
| timesfm | spe_berg       | 7       | per_well   | well_14 | TimesFM is not installed. Requires Python <3.12 and JAX. Install via: pip install timesfm (in a Python 3.11 environment) |
| tirex   | spe_berg       | 7       | per_well   | well_14 | TiRex is not installed. Requires GPU with CUDA ≥8.0. Install via: pip install git+https://github.com/NX-AI/tirex         |
| timesfm | spe_berg       | 7       | per_well   | well_15 | TimesFM is not installed. Requires Python <3.12 and JAX. Install via: pip install timesfm (in a Python 3.11 environment) |
| tirex   | spe_berg       | 7       | per_well   | well_15 | TiRex is not installed. Requires GPU with CUDA ≥8.0. Install via: pip install git+https://github.com/NX-AI/tirex         |
| timesfm | spe_berg       | 7       | per_well   | well_16 | TimesFM is not installed. Requires Python <3.12 and JAX. Install via: pip install timesfm (in a Python 3.11 environment) |
| tirex   | spe_berg       | 7       | per_well   | well_16 | TiRex is not installed. Requires GPU with CUDA ≥8.0. Install via: pip install git+https://github.com/NX-AI/tirex         |
| timesfm | spe_berg       | 7       | per_well   | well_17 | TimesFM is not installed. Requires Python <3.12 and JAX. Install via: pip install timesfm (in a Python 3.11 environment) |
| tirex   | spe_berg       | 7       | per_well   | well_17 | TiRex is not installed. Requires GPU with CUDA ≥8.0. Install via: pip install git+https://github.com/NX-AI/tirex         |
| timesfm | spe_berg       | 7       | per_well   | well_18 | TimesFM is not installed. Requires Python <3.12 and JAX. Install via: pip install timesfm (in a Python 3.11 environment) |
| tirex   | spe_berg       | 7       | per_well   | well_18 | TiRex is not installed. Requires GPU with CUDA ≥8.0. Install via: pip install git+https://github.com/NX-AI/tirex         |
| timesfm | spe_berg       | 7       | per_well   | well_19 | TimesFM is not installed. Requires Python <3.12 and JAX. Install via: pip install timesfm (in a Python 3.11 environment) |
| tirex   | spe_berg       | 7       | per_well   | well_19 | TiRex is not installed. Requires GPU with CUDA ≥8.0. Install via: pip install git+https://github.com/NX-AI/tirex         |
| timesfm | spe_berg       | 7       | per_well   | well_20 | TimesFM is not installed. Requires Python <3.12 and JAX. Install via: pip install timesfm (in a Python 3.11 environment) |
| tirex   | spe_berg       | 7       | per_well   | well_20 | TiRex is not installed. Requires GPU with CUDA ≥8.0. Install via: pip install git+https://github.com/NX-AI/tirex         |
| timesfm | spe_berg       | 7       | per_well   | well_21 | TimesFM is not installed. Requires Python <3.12 and JAX. Install via: pip install timesfm (in a Python 3.11 environment) |
| tirex   | spe_berg       | 7       | per_well   | well_21 | TiRex is not installed. Requires GPU with CUDA ≥8.0. Install via: pip install git+https://github.com/NX-AI/tirex         |

## Notes

- Expected grid is built from the four forecasting data configs and the discovered/default forecasting model set.
- `missing` means a model × dataset × horizon × mode × well/scenario artifact is expected by config but no aggregate-ingestable JSON currently exists.
- `unavailable` means an artifact exists but records dependency/runtime unavailability rather than metrics.
