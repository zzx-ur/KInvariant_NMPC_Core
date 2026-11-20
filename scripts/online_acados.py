#!/usr/bin/env python3
"""Run one online K-step invariant NMPC update with acados.

The offline symbolic controller uses the paper's one-second discrete model.
The online OCP uses exactly that same map and the paper's feasible
initialization strategy.  Solver construction/code generation is deliberately
outside the reported online timing.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import casadi as ca
import numpy as np
from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver
from scipy.io import loadmat


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOOKUP_FILE = PROJECT_ROOT / "data" / "offline" / "paper_k5_lookup.mat"
CODE_EXPORT_DIRECTORY = PROJECT_ROOT / ".cache" / "acados_online_k5_n10"

OFFLINE_STEP_S = 1.0
PREDICTION_STEPS = 10
SQP_MAX_ITERATIONS = 20
INITIAL_STATE = np.array([1.0, -1.3, -0.7854])
REFERENCE_POSITION = np.array([0.5, 0.5])
ACADOS_INFINITY = 1.0e10
FEASIBILITY_TOLERANCE = 1.0e-5


@dataclass(frozen=True)
class OfflineLookup:
    k: int
    lower_terminal: np.ndarray
    upper_terminal: np.ndarray
    state_bounds: np.ndarray
    input_bounds: np.ndarray
    cell_width: np.ndarray
    cell_count: np.ndarray
    strides: np.ndarray
    modes: np.ndarray
    controller_safe: np.ndarray
    controller_reach: np.ndarray
    controllable: np.ndarray

    @classmethod
    def load(cls, path: Path) -> "OfflineLookup":
        data = loadmat(path)
        return cls(
            k=int(data["K"].item()),
            lower_terminal=data["lb"].reshape(-1).astype(float),
            upper_terminal=data["ub"].reshape(-1).astype(float),
            state_bounds=data["bound"].astype(float),
            input_bounds=data["inputbound"].astype(float),
            cell_width=data["d_x"].reshape(-1).astype(float),
            cell_count=data["n_x"].reshape(-1).astype(int),
            strides=data["p_x"].reshape(-1).astype(int),
            modes=data["mode"].astype(float),
            controller_safe=data["Controller1"].reshape(-1).astype(int),
            controller_reach=data["Controller2"].astype(int),
            controllable=data["cpre_k_set"].reshape(-1).astype(bool),
        )

    def state_index(self, state: np.ndarray) -> int:
        wrapped = np.asarray(state, dtype=float).copy()
        wrapped[2] = wrap_angle(wrapped[2])
        coordinates = np.floor(
            (wrapped - self.state_bounds[:, 0]) / self.cell_width
        ).astype(int)
        coordinates = np.clip(coordinates, 0, self.cell_count - 1)
        return int(coordinates @ self.strides[:3])

    def control(self, state: np.ndarray, *, reach_terminal: bool) -> np.ndarray:
        state_index = self.state_index(state)
        if not self.controllable[state_index]:
            raise RuntimeError("The initialization trajectory left CPre[1:K](T).")

        if reach_terminal:
            available = self.controller_reach[state_index]
            nonzero = np.flatnonzero(available)
            if nonzero.size == 0:
                raise RuntimeError("No offline terminal-reaching control is available.")
            mode_index = int(available[nonzero[0]])
        else:
            mode_index = int(self.controller_safe[state_index])

        if mode_index == 0:
            raise RuntimeError("No offline safety control is available.")
        return self.modes[:, mode_index - 1].copy()


@dataclass(frozen=True)
class OnlineStepResult:
    elapsed_seconds: float
    terminal_stage: int
    cost: float
    controls: np.ndarray
    states: np.ndarray
    attempted_stages: tuple[int, ...]
    solver_statuses: tuple[int, ...]
    acceptance: str
    max_constraint_violation: float


def wrap_angle(angle: float) -> float:
    return (float(angle) + np.pi) % (2.0 * np.pi) - np.pi


def offline_endpoint(state: np.ndarray, control: np.ndarray) -> np.ndarray:
    """One-second discrete endpoint used to build the supplied lookup."""
    state = np.asarray(state, dtype=float)
    speed, yaw_rate = np.asarray(control, dtype=float)
    return state + OFFLINE_STEP_S * np.array(
        [speed * np.cos(state[2]), speed * np.sin(state[2]), yaw_rate]
    )


def offline_initial_sequence(
    state: np.ndarray, lookup: OfflineLookup
) -> tuple[np.ndarray, np.ndarray, int]:
    """Implement the paper's initial-time feasible sequence (Algorithm 1)."""
    total_steps = PREDICTION_STEPS + lookup.k - 1
    controls: list[np.ndarray] = []
    states = [np.asarray(state, dtype=float).copy()]

    # This indexing matches the supplied MATLAB implementation: maintain the
    # CPre union for Ns-1 steps, then reach T in at most K steps.
    for _ in range(total_steps - lookup.k):
        control = lookup.control(states[-1], reach_terminal=False)
        controls.append(control)
        states.append(offline_endpoint(states[-1], control))

    start_reach_stage = len(controls)
    state_index = lookup.state_index(states[-1])
    available = np.flatnonzero(lookup.controller_reach[state_index])
    if available.size == 0:
        raise RuntimeError("Offline lookup cannot connect the initial sequence to T.")
    reach_steps = int(available[0] + 1)

    for _ in range(reach_steps):
        control = lookup.control(states[-1], reach_terminal=True)
        controls.append(control)
        states.append(offline_endpoint(states[-1], control))

    terminal_stage = start_reach_stage + reach_steps
    while len(controls) < total_steps:
        control = lookup.control(states[-1], reach_terminal=False)
        controls.append(control)
        states.append(offline_endpoint(states[-1], control))

    control_array = np.asarray(controls)
    state_array = np.asarray(states)
    if control_array.shape != (total_steps, 2):
        raise RuntimeError("Offline initialization returned an invalid sequence length.")
    if not np.all(
        (state_array[terminal_stage] >= lookup.lower_terminal - 1.0e-12)
        & (state_array[terminal_stage] <= lookup.upper_terminal + 1.0e-12)
    ):
        raise RuntimeError("Offline initialization did not reach terminal set T.")
    return control_array, state_array, terminal_stage


def rollout_controls(state: np.ndarray, controls: np.ndarray) -> np.ndarray:
    """Roll out a control sequence with the lookup table's one-second model."""
    states = [np.asarray(state, dtype=float).copy()]
    for control in controls:
        states.append(offline_endpoint(states[-1], control))
    return np.asarray(states)


def trajectory_feasibility_violation(
    state: np.ndarray,
    controls: np.ndarray,
    terminal_stage: int,
    lookup: OfflineLookup,
) -> tuple[np.ndarray, float]:
    """Certify a candidate by exact rollout, independently of SQP convergence."""
    controls = np.asarray(controls, dtype=float)
    total_steps = PREDICTION_STEPS + lookup.k - 1
    if controls.shape != (total_steps, 2) or not np.all(np.isfinite(controls)):
        return np.empty((0, 3)), np.inf
    if terminal_stage not in range(PREDICTION_STEPS, total_steps + 1):
        return np.empty((0, 3)), np.inf

    states = rollout_controls(state, controls)
    if not np.all(np.isfinite(states)):
        return states, np.inf

    input_violation = np.max(
        np.maximum(
            lookup.input_bounds[:, 0] - controls,
            controls - lookup.input_bounds[:, 1],
        )
    )
    safe_violation = max(
        float(np.max(states[:, 0] ** 2 - states[:, 1] ** 2 - 4.0)),
        float(np.max(4.0 * states[:, 1] ** 2 - states[:, 0] ** 2 - 16.0)),
    )
    terminal_position = states[terminal_stage, :2]
    terminal_violation = np.max(
        np.maximum(
            lookup.lower_terminal[:2] - terminal_position,
            terminal_position - lookup.upper_terminal[:2],
        )
    )
    return states, max(
        float(input_violation),
        safe_violation,
        float(terminal_violation),
        0.0,
    )


def trajectory_cost(
    states: np.ndarray,
    controls: np.ndarray,
    reference_position: np.ndarray,
) -> float:
    """Evaluate the same finite-horizon cost used by the acados OCP."""
    reference_position = np.asarray(reference_position, dtype=float)
    running_position_error = (
        states[:PREDICTION_STEPS, :2] - reference_position
    )
    terminal_position_error = (
        states[PREDICTION_STEPS, :2] - reference_position
    )
    return float(
        100.0 * np.sum(running_position_error**2)
        + np.sum(controls[:PREDICTION_STEPS] ** 2)
        + 100.0 * np.sum(terminal_position_error**2)
    )


def sliding_initial_sequence(
    state: np.ndarray,
    previous_solution: OnlineStepResult,
    lookup: OfflineLookup,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Shift the solution and append the offline backup controller (Algorithm 5)."""
    total_steps = previous_solution.controls.shape[0]
    terminal_stage = previous_solution.terminal_stage
    candidate_stages = range(PREDICTION_STEPS, total_steps + 1)
    if terminal_stage not in candidate_stages:
        raise ValueError(f"Invalid previous terminal stage {terminal_stage}.")

    # After applying u[0], controls u[1], ..., u[terminal_stage-1]
    # lead to the previous terminal state at new stage terminal_stage-1.
    controls = [
        control.copy()
        for control in previous_solution.controls[1:terminal_stage]
    ]
    states = list(rollout_controls(state, np.asarray(controls)))

    if terminal_stage == PREDICTION_STEPS:
        # The shifted terminal state is now at stage Ns-1, so use Controller2
        # to return to T in at most K steps, exactly as in the MATLAB routine.
        state_index = lookup.state_index(states[-1])
        available = np.flatnonzero(lookup.controller_reach[state_index])
        if available.size == 0:
            raise RuntimeError("The shifted state has no terminal-reaching control.")
        reach_steps = int(available[0] + 1)
        for _ in range(reach_steps):
            control = lookup.control(states[-1], reach_terminal=True)
            controls.append(control)
            states.append(offline_endpoint(states[-1], control))
        preferred_terminal_stage = len(controls)
    else:
        preferred_terminal_stage = terminal_stage - 1

    while len(controls) < total_steps:
        control = lookup.control(states[-1], reach_terminal=False)
        controls.append(control)
        states.append(offline_endpoint(states[-1], control))

    control_array = np.asarray(controls)
    state_array = np.asarray(states)
    terminal_position = state_array[preferred_terminal_stage, :2]
    if control_array.shape != (total_steps, 2):
        raise RuntimeError("Sliding initialization returned an invalid sequence length.")
    terminal_violation = float(
        np.max(
            np.maximum(
                lookup.lower_terminal[:2] - terminal_position,
                terminal_position - lookup.upper_terminal[:2],
            )
        )
    )
    if terminal_violation > FEASIBILITY_TOLERANCE:
        raise RuntimeError(
            "Sliding initialization did not return to terminal set T: "
            f"violation={terminal_violation:.3e}."
        )
    return control_array, state_array, preferred_terminal_stage


def build_ocp(
    lookup: OfflineLookup, reference_position: np.ndarray | None = None
) -> AcadosOcp:
    total_steps = PREDICTION_STEPS + lookup.k - 1
    if reference_position is None:
        reference_position = REFERENCE_POSITION
    reference_position = np.asarray(reference_position, dtype=float)
    if reference_position.shape != (2,):
        raise ValueError("reference_position must contain exactly two values.")

    x = ca.SX.sym("x", 3)
    u = ca.SX.sym("u", 2)

    # Keep the online dynamics identical to the one-second discrete map used
    # to create paper_k5_lookup.mat.
    x_next = ca.vertcat(
        x[0] + OFFLINE_STEP_S * u[0] * ca.cos(x[2]),
        x[1] + OFFLINE_STEP_S * u[0] * ca.sin(x[2]),
        x[2] + OFFLINE_STEP_S * u[1],
    )

    model = AcadosModel()
    model.name = "kinv_online_k5_n10"
    model.x = x
    model.u = u
    model.disc_dyn_expr = x_next

    position_and_input = ca.vertcat(x[0], x[1], u[0], u[1])
    safe_and_terminal = ca.vertcat(
        x[0] ** 2 - x[1] ** 2,
        4.0 * x[1] ** 2 - x[0] ** 2,
        x[0],
        x[1],
    )
    model.cost_y_expr = position_and_input
    model.cost_y_expr_e = x
    model.con_h_expr = safe_and_terminal
    model.con_h_expr_e = safe_and_terminal

    ocp = AcadosOcp()
    ocp.model = model
    ocp.solver_options.N_horizon = total_steps
    ocp.solver_options.tf = total_steps * OFFLINE_STEP_S
    ocp.solver_options.integrator_type = "DISCRETE"

    ocp.cost.cost_type = "NONLINEAR_LS"
    ocp.cost.cost_type_e = "NONLINEAR_LS"
    ocp.cost.W = 2.0 * np.diag([100.0, 100.0, 1.0, 1.0])
    ocp.cost.W_e = np.zeros((3, 3))
    ocp.cost.yref = np.r_[reference_position, 0.0, 0.0]
    ocp.cost.yref_e = np.zeros(3)

    ocp.constraints.x0 = INITIAL_STATE
    ocp.constraints.idxbu = np.array([0, 1])
    ocp.constraints.lbu = lookup.input_bounds[:, 0]
    ocp.constraints.ubu = lookup.input_bounds[:, 1]
    safe_lower = np.array([-ACADOS_INFINITY] * 4)
    safe_upper = np.array([4.0, 16.0, ACADOS_INFINITY, ACADOS_INFINITY])
    ocp.constraints.lh = safe_lower
    ocp.constraints.uh = safe_upper
    ocp.constraints.lh_e = safe_lower
    ocp.constraints.uh_e = safe_upper

    ocp.solver_options.nlp_solver_type = "SQP"
    ocp.solver_options.nlp_solver_max_iter = SQP_MAX_ITERATIONS
    ocp.solver_options.hessian_approx = "EXACT"
    ocp.solver_options.regularize_method = "CONVEXIFY"
    ocp.solver_options.globalization = "MERIT_BACKTRACKING"
    ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
    ocp.solver_options.qp_solver_cond_N = total_steps
    ocp.solver_options.tol = 1.0e-5
    ocp.solver_options.print_level = 0
    ocp.code_gen_options.code_export_directory = str(CODE_EXPORT_DIRECTORY)
    ocp.code_gen_options.json_file = "kinv_online.json"
    return ocp


def set_stage_costs(solver: AcadosOcpSolver, total_steps: int) -> None:
    running_weight = 2.0 * np.diag([100.0, 100.0, 1.0, 1.0])
    terminal_weight = 2.0 * np.diag([100.0, 100.0, 0.0, 0.0])
    zero_weight = np.zeros((4, 4))
    for stage in range(total_steps):
        if stage < PREDICTION_STEPS:
            weight = running_weight
        elif stage == PREDICTION_STEPS:
            weight = terminal_weight
        else:
            weight = zero_weight
        solver.cost_set(stage, "W", weight)


def set_terminal_candidate(
    solver: AcadosOcpSolver,
    terminal_stage: int,
    total_steps: int,
    lookup: OfflineLookup,
) -> None:
    safe_lower = np.array([-ACADOS_INFINITY] * 4)
    safe_upper = np.array([4.0, 16.0, ACADOS_INFINITY, ACADOS_INFINITY])
    terminal_lower = safe_lower.copy()
    terminal_upper = safe_upper.copy()
    terminal_lower[2:] = lookup.lower_terminal[:2]
    terminal_upper[2:] = lookup.upper_terminal[:2]

    for stage in range(1, total_steps + 1):
        lower = terminal_lower if stage == terminal_stage else safe_lower
        upper = terminal_upper if stage == terminal_stage else safe_upper
        solver.constraints_set(stage, "lh", lower)
        solver.constraints_set(stage, "uh", upper)


def set_initial_guess(
    solver: AcadosOcpSolver, controls: np.ndarray, states: np.ndarray
) -> None:
    for stage, control in enumerate(controls):
        solver.set(stage, "x", states[stage])
        solver.set(stage, "u", control)
    solver.set(len(controls), "x", states[-1])


def solve_one_online_step(
    solver: AcadosOcpSolver,
    state: np.ndarray,
    controls: np.ndarray,
    states: np.ndarray,
    lookup: OfflineLookup,
    preferred_terminal_stage: int,
    reference_position: np.ndarray | None = None,
) -> OnlineStepResult:
    """Return the first certified feasible iterate, with the backup as fallback."""
    total_steps = controls.shape[0]
    if reference_position is None:
        reference_position = REFERENCE_POSITION
    valid_candidate_stages = list(range(PREDICTION_STEPS, total_steps + 1))
    if preferred_terminal_stage not in valid_candidate_stages:
        raise ValueError(
            f"Preferred terminal stage {preferred_terminal_stage} is outside "
            f"the candidate range {valid_candidate_stages}."
        )
    # Recursive feasibility is the priority: spend a bounded amount of time on
    # the offline-predicted stage, then use the already certified backup rather
    # than solving four additional terminal-stage candidates.
    candidate_stages = [preferred_terminal_stage]
    attempted_stages: list[int] = []
    solver_statuses: list[int] = []

    started = perf_counter()
    backup_states, backup_violation = trajectory_feasibility_violation(
        state, controls, preferred_terminal_stage, lookup
    )
    if backup_violation > FEASIBILITY_TOLERANCE:
        raise RuntimeError(
            "The offline/sliding initialization is not a feasible backup: "
            f"violation={backup_violation:.3e}."
        )

    for terminal_stage in candidate_stages:
        solver.reset()
        solver.constraints_set(0, "lbx", state)
        solver.constraints_set(0, "ubx", state)
        set_initial_guess(solver, controls, states)
        set_terminal_candidate(solver, terminal_stage, total_steps, lookup)
        status = solver.solve()
        attempted_stages.append(terminal_stage)
        solver_statuses.append(status)
        solved_controls = np.vstack(
            [solver.get(stage, "u") for stage in range(total_steps)]
        )
        solved_states, solved_violation = trajectory_feasibility_violation(
            state, solved_controls, terminal_stage, lookup
        )
        if solved_violation <= FEASIBILITY_TOLERANCE:
            return OnlineStepResult(
                elapsed_seconds=perf_counter() - started,
                terminal_stage=terminal_stage,
                cost=trajectory_cost(
                    solved_states, solved_controls, reference_position
                ),
                controls=solved_controls,
                states=solved_states,
                attempted_stages=tuple(attempted_stages),
                solver_statuses=tuple(solver_statuses),
                acceptance="converged" if status == 0 else "feasible_iterate",
                max_constraint_violation=solved_violation,
            )

    return OnlineStepResult(
        elapsed_seconds=perf_counter() - started,
        terminal_stage=preferred_terminal_stage,
        cost=trajectory_cost(backup_states, controls, reference_position),
        controls=np.asarray(controls).copy(),
        states=backup_states,
        attempted_stages=tuple(attempted_stages),
        solver_statuses=tuple(solver_statuses),
        acceptance="certified_backup",
        max_constraint_violation=backup_violation,
    )


def main() -> None:
    lookup = OfflineLookup.load(LOOKUP_FILE)
    if lookup.k != 5:
        raise RuntimeError(f"Expected K=5 in {LOOKUP_FILE}, got K={lookup.k}.")

    controls, states, preferred_terminal_stage = offline_initial_sequence(
        INITIAL_STATE, lookup
    )
    ocp = build_ocp(lookup)
    solver = AcadosOcpSolver(ocp, verbose=False)
    set_stage_costs(solver, controls.shape[0])

    result = solve_one_online_step(
        solver,
        INITIAL_STATE,
        controls,
        states,
        lookup,
        preferred_terminal_stage,
    )
    print(f"{result.elapsed_seconds * 1.0e3:.3f} ms")


if __name__ == "__main__":
    main()
