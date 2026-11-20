#!/usr/bin/env python3
"""Run a 50-second closed-loop SQP test at the safety-boundary reference."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from acados_template import AcadosOcpSolver
from matplotlib.patches import Rectangle

from online_acados import (
    INITIAL_STATE,
    LOOKUP_FILE,
    PROJECT_ROOT,
    SQP_MAX_ITERATIONS,
    OfflineLookup,
    build_ocp,
    offline_endpoint,
    offline_initial_sequence,
    set_stage_costs,
    sliding_initial_sequence,
    solve_one_online_step,
    wrap_angle,
)


SIMULATION_STEPS = 50
HARD_REFERENCE_POSITION = np.array([np.sqrt(32.0 / 3.0), np.sqrt(20.0 / 3.0)])


def safety_violation(state: np.ndarray) -> float:
    x_position, y_position = state[:2]
    return max(
        x_position**2 - y_position**2 - 4.0,
        4.0 * y_position**2 - x_position**2 - 16.0,
        0.0,
    )


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=SIMULATION_STEPS)
    parser.add_argument(
        "--max-sqp-iterations", type=int, default=SQP_MAX_ITERATIONS
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "results" / "hard_reference_trajectory.png",
    )
    return parser.parse_args()


def plot_trajectory(
    states: np.ndarray,
    lookup: OfflineLookup,
    output_path: Path,
) -> None:
    x_grid = np.linspace(-3.6, 3.6, 500)
    y_grid = np.linspace(-3.0, 3.0, 500)
    grid_x, grid_y = np.meshgrid(x_grid, y_grid)
    constraint_1 = grid_x**2 - grid_y**2 - 4.0
    constraint_2 = 4.0 * grid_y**2 - grid_x**2 - 16.0
    safe_region = (constraint_1 <= 0.0) & (constraint_2 <= 0.0)

    figure, axes = plt.subplots(figsize=(7.2, 6.2), constrained_layout=True)
    axes.contourf(
        grid_x,
        grid_y,
        safe_region.astype(float),
        levels=[0.5, 1.5],
        colors=["#dcefe2"],
        alpha=0.75,
    )
    first_boundary = axes.contour(
        grid_x,
        grid_y,
        constraint_1,
        levels=[0.0],
        colors=["#3268a8"],
        linewidths=1.8,
    )
    second_boundary = axes.contour(
        grid_x,
        grid_y,
        constraint_2,
        levels=[0.0],
        colors=["#a35424"],
        linewidths=1.8,
    )
    trajectory_line = axes.plot(
        states[:, 0],
        states[:, 1],
        color="#151515",
        linewidth=1.8,
        marker="o",
        markersize=2.8,
        label="closed-loop trajectory",
    )[0]
    start_marker = axes.scatter(
        states[0, 0],
        states[0, 1],
        s=60,
        marker="o",
        color="#247a3b",
        zorder=5,
        label="initial state",
    )
    reference_marker = axes.scatter(
        HARD_REFERENCE_POSITION[0],
        HARD_REFERENCE_POSITION[1],
        s=110,
        marker="*",
        color="#b6252a",
        zorder=6,
        label="boundary reference",
    )
    terminal_rectangle = Rectangle(
        lookup.lower_terminal[:2],
        *(lookup.upper_terminal[:2] - lookup.lower_terminal[:2]),
        fill=False,
        edgecolor="#6a4c93",
        linewidth=1.4,
        linestyle="--",
        label="terminal set T",
    )
    axes.add_patch(terminal_rectangle)

    axes.set_xlim(x_grid[0], x_grid[-1])
    axes.set_ylim(y_grid[0], y_grid[-1])
    axes.set_aspect("equal", adjustable="box")
    axes.set_xlabel(r"$x_1$")
    axes.set_ylabel(r"$x_2$")
    axes.set_title("K-invariant NMPC: hard boundary reference")
    axes.grid(True, linewidth=0.5, alpha=0.3)
    axes.legend(
        [
            first_boundary.legend_elements()[0][0],
            second_boundary.legend_elements()[0][0],
            trajectory_line,
            start_marker,
            reference_marker,
            terminal_rectangle,
        ],
        [
            r"$x_1^2-x_2^2=4$",
            r"$4x_2^2-x_1^2=16$",
            "closed-loop trajectory",
            "initial state",
            "boundary reference",
            "terminal set T",
        ],
        loc="lower left",
        fontsize=8,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def main() -> None:
    arguments = parse_arguments()
    if arguments.steps <= 0 or arguments.max_sqp_iterations <= 0:
        raise ValueError("steps and max-sqp-iterations must be positive.")

    lookup = OfflineLookup.load(LOOKUP_FILE)
    controls, states, preferred_stage = offline_initial_sequence(
        INITIAL_STATE, lookup
    )
    ocp = build_ocp(lookup, HARD_REFERENCE_POSITION)
    ocp.model.name = (
        f"kinv_online_k5_n10_hard_sqp_{arguments.max_sqp_iterations}"
    )
    ocp.solver_options.nlp_solver_max_iter = arguments.max_sqp_iterations
    ocp.code_gen_options.code_export_directory = str(
        PROJECT_ROOT
        / ".cache"
        / f"acados_online_hard_sqp_{arguments.max_sqp_iterations}"
    )
    ocp.code_gen_options.json_file = (
        f"kinv_online_hard_sqp_{arguments.max_sqp_iterations}.json"
    )
    solver = AcadosOcpSolver(ocp, verbose=False)
    set_stage_costs(solver, controls.shape[0])

    state = INITIAL_STATE.copy()
    closed_loop_states = [state.copy()]
    solve_times_ms: list[float] = []
    candidate_attempts: list[int] = []
    fallback_steps: list[int] = []
    fallback_details: list[str] = []
    acceptance_counts = {
        "converged": 0,
        "feasible_iterate": 0,
        "certified_backup": 0,
    }
    certified_backup_steps: list[int] = []
    costs: list[float] = []
    failure: str | None = None
    failure_state: np.ndarray | None = None
    failure_guess_violation: float | None = None

    for step in range(1, arguments.steps + 1):
        guess_safe_violation = max(safety_violation(x) for x in states)
        guess_terminal_position = states[preferred_stage, :2]
        guess_terminal_violation = max(
            float(
                np.max(
                    np.maximum(
                        lookup.lower_terminal[:2] - guess_terminal_position,
                        guess_terminal_position - lookup.upper_terminal[:2],
                    )
                )
            ),
            0.0,
        )
        guess_input_violation = max(
            float(
                np.max(
                    np.maximum(
                        lookup.input_bounds[:, 0] - controls,
                        controls - lookup.input_bounds[:, 1],
                    )
                )
            ),
            0.0,
        )
        current_guess_violation = max(
            guess_safe_violation,
            guess_terminal_violation,
            guess_input_violation,
        )
        try:
            result = solve_one_online_step(
                solver,
                state,
                controls,
                states,
                lookup,
                preferred_stage,
                HARD_REFERENCE_POSITION,
            )
        except RuntimeError as error:
            failure = f"step {step}: online solve failed: {error}"
            failure_state = state.copy()
            failure_guess_violation = current_guess_violation
            break

        solve_times_ms.append(result.elapsed_seconds * 1.0e3)
        costs.append(result.cost)
        acceptance_counts[result.acceptance] += 1
        if result.acceptance == "certified_backup":
            certified_backup_steps.append(step)
        candidate_attempts.append(len(result.attempted_stages))
        if len(result.attempted_stages) > 1:
            fallback_steps.append(step)
            fallback_details.append(
                f"step {step}: "
                + ", ".join(
                    f"stage {stage}/status {status}"
                    for stage, status in zip(
                        result.attempted_stages, result.solver_statuses
                    )
                )
            )

        state = offline_endpoint(state, result.controls[0])
        state[2] = wrap_angle(state[2])
        closed_loop_states.append(state.copy())

        if safety_violation(state) > 1.0e-5:
            failure = (
                f"step {step}: applied state violates the safe set by "
                f"{safety_violation(state):.3e}"
            )
            break

        try:
            controls, states, preferred_stage = sliding_initial_sequence(
                state, result, lookup
            )
        except RuntimeError as error:
            failure = f"step {step}: sliding initialization failed: {error}"
            break

    state_array = np.asarray(closed_loop_states)
    solved_steps = len(solve_times_ms)
    final_distance = float(
        np.linalg.norm(state_array[-1, :2] - HARD_REFERENCE_POSITION)
    )
    maximum_safety_violation = max(safety_violation(x) for x in state_array)
    plot_trajectory(state_array, lookup, arguments.output)

    print(f"reference={HARD_REFERENCE_POSITION}")
    print(f"initial_state={INITIAL_STATE}")
    print(f"max_sqp_iterations={arguments.max_sqp_iterations}")
    print(f"solved_steps={solved_steps}/{arguments.steps}")
    print(f"failure={failure or 'none'}")
    if failure_state is not None:
        print(f"failure_state={failure_state}")
        print(f"failure_preferred_stage={preferred_stage}")
        print(f"failure_initial_guess_violation={failure_guess_violation:.3e}")
    print(f"fallback_steps={fallback_steps}")
    print(f"fallback_details={fallback_details}")
    print(f"acceptance_counts={acceptance_counts}")
    print(f"certified_backup_steps={certified_backup_steps}")
    print(f"max_candidate_attempts={max(candidate_attempts, default=0)}")
    if solve_times_ms:
        print(
            "solve_time_ms="
            f"mean:{np.mean(solve_times_ms):.3f}, "
            f"max:{np.max(solve_times_ms):.3f}, "
            f"p95:{np.percentile(solve_times_ms, 95):.3f}"
        )
        print(
            "cost="
            f"first:{costs[0]:.6f}, final:{costs[-1]:.6f}, "
            f"max:{np.max(costs):.6f}, "
            f"increasing_steps:{sum(b > a for a, b in zip(costs, costs[1:]))}"
        )
    print(f"final_state={state_array[-1]}")
    print(f"final_position_error={final_distance:.6f}")
    print(f"max_closed_loop_safety_violation={maximum_safety_violation:.3e}")
    print(f"trajectory_plot={arguments.output.resolve()}")


if __name__ == "__main__":
    main()
