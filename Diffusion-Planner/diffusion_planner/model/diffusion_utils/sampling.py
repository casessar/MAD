from typing import Dict
import torch
import diffusion_planner.model.diffusion_utils.dpm_solver_pytorch as dpm


def dpm_sampler(
        model: torch.nn.Module,
        x_T,
        other_model_params: Dict = {},
        diffusion_steps: int = 10,
        noise_schedule_params: Dict = {},
        model_wrapper_params: Dict = {},
        dpm_solver_params: Dict = {},
        sample_params: Dict = {},
        # Truncated Diffusion：从 t_start（0~1）开始去噪，而非从 t=1.0
        # 设为 None 时退化为标准全程扩散
        t_start: float = None,
    ):
    """
    DPM-Solver 采样器。

    Args:
        t_start: 截断起始时刻（0~1），对应 dpm_solver.sample 的 t_T 参数。
                 None 表示标准全程扩散（t_T=1.0）。
                 推荐值：0.3~0.7，越小去噪步数越少，保留 anchor 信息越多。
    """
    with torch.no_grad():
        noise_schedule = dpm.NoiseScheduleVP(
            schedule='linear',
            **noise_schedule_params
        )

        model_fn = dpm.model_wrapper(
            model,
            noise_schedule,
            model_type=model.model_type,
            model_kwargs=other_model_params,
            **model_wrapper_params
        )

        dpm_solver = dpm.DPM_Solver(
            model_fn, noise_schedule, algorithm_type="dpmsolver++",
            **dpm_solver_params
        )

        # t_T：去噪起始时刻；t_start 覆盖默认的 1.0
        t_T = t_start if t_start is not None else 1.0

        # sample_params 中也可能传了旧的 t_start key，需要过滤掉避免重复
        clean_sample_params = {k: v for k, v in sample_params.items() if k != 't_start'}

        sample_dpm = dpm_solver.sample(
            x_T,
            steps=diffusion_steps,
            order=2,
            skip_type="logSNR",
            method="multistep",
            denoise_to_zero=True,
            t_T=t_T,
            **clean_sample_params
        )

    return sample_dpm
