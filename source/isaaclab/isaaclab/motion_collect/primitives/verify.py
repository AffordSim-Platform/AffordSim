"""Recheck the complete arrangement after release, including earlier objects."""
from ..success.relations import evaluate_scene_relation
from .utils import get_prev_alive_mask, commit_task_failures


def primitive_verify(collector, mode, params, context):
    del mode
    rules = params.get("relations", [])
    if not rules:
        raise ValueError("verify requires at least one geometric relation")
    n = collector.num_envs
    alive = get_prev_alive_mask(context, n)
    passed = [True] * n
    history = [[] for _ in range(n)]
    for _ in range(int(params.get("samples", 3))):
        collector.control_gripper_batch([1.0 if s else 0.0 for s in alive],
                                        steps=int(params.get("settle_steps", 4)), active_mask=alive)
        for i in range(n):
            sample = []
            for rule in rules:
                ok, metrics = evaluate_scene_relation(collector, i, rule)
                passed[i] = passed[i] and ok
                sample.append({"source": rule["source"], "target": rule["target"], **metrics})
            history[i].append(sample)
    commit_task_failures(context, alive, alive, passed)
    final = [bool(a and b) for a, b in zip(alive, passed)]
    context["env_alive_mask"] = final
    context["post_verify_success_mask"] = final
    context["task_success_metrics"] = [{"final_arrangement_valid": p, "samples": h}
                                       for p, h in zip(passed, history)]
    context["primitive_executed_motion"] = True
    print(f"   [Primitive] Final arrangement: {sum(final)}/{n}; {history}")
    return any(final)
