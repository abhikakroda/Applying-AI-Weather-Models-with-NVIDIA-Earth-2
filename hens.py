# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import numpy as np
import torch
import xarray as xr
from earth2studio.models.px import PrognosticModel


def get_noise_vector(
    model: PrognosticModel,
    skill_path: str | None = None,
    noise_amplification: float = 1.0,
    perturbed_var: str | list[str] | None = None,
    lead_time: int = 48,
) -> torch.Tensor:
    """Generate a noise vector for the HemisphericCentredBredVector perturbation method.

    This function creates a variable-specific noise vector based on forecast skill scores.
    The noise amplitude for each variable is scaled according to its forecast skill at the
    specified lead time, with variables not in the perturbation list set to zero.

    Parameters
    ----------
    model : PrognosticModel
        The prognostic model used for forecasting
    skill_path : str, optional
        Path to the file containing model skill scores (RMSE/MSE), by default None
    noise_amplification : float, optional
        Base amplification factor for the noise vector, by default 1.0
    perturbed_var : str | list[str] | None, optional
        Variables to be perturbed. If None, all model variables are perturbed, by default None
    lead_time : int, optional
        Lead time at which to evaluate model skill, by default 48

    Returns
    -------
    torch.Tensor
        A noise vector with shape (1, 1, 1, n_vars, 1, 1) where n_vars is the number of model variables

    Raises
    ------
    ValueError
        If skill_path is not provided
    """
    if skill_path is None:
        raise ValueError(
            f"provide path to data set containing {lead_time}h deterministic [r]mse"
        )

    model_vars = model.input_coords()["variable"]
    if perturbed_var is None:
        perturbed_var = model_vars
    elif isinstance(perturbed_var, str):
        perturbed_var = [perturbed_var]

    # set noise for variables which shall not be perturbed to 0.
    skill = xr.open_dataset(skill_path)
    scale_vec = torch.Tensor(
        np.asarray(
            [
                (
                    skill.sel(channel=var, lead_time=lead_time)["value"].item()
                    if var in perturbed_var
                    else 0.0
                )
                for var in model_vars
            ]
        )
    )

    return scale_vec.reshape(1, 1, 1, -1, 1, 1) * noise_amplification
