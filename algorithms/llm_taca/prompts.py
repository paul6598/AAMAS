"""Prompt construction for LLM-TACA (paper Sec. 3.3): p^T_base = (p_env, p^T_desc, p_defn,
p^T_task). Extends LLM-MCA's task query to also request, per agent, an explicit task assignment.

The task is a RECOMMENDED ACTION (a single integer in 0..n_actions-1) -- a short integer array
as in the paper, and env-agnostic (works for the matrix game and for foraging alike), unlike a
grid-specific target cell.
"""
from ..llm_mca.prompts import build_defn_prompt, build_critic_prompt  # noqa: F401 (re-export)


def build_task_prompt(agent_names, num_timesteps, n_actions):
    """p^T_task: request per-timestep credit arrays AND a per-agent recommended action."""
    names = list(agent_names)
    name_list = " and ".join([", ".join(names[:-1]), names[-1]]) if len(names) > 1 else names[0]
    credit_sentences = " ".join(
        f"Along with your explanation, create a numpy array called {n.lower()}_credit with "
        f"numerical credit values for {n}." for n in names
    )
    task_sentences = " ".join(
        f"Also create a variable called task_{n.lower()} set to the single action index "
        f"(an integer from 0 to {n_actions - 1}) that you think {n} should take next to help the "
        f"team; if you have no recommendation, set task_{n.lower()} = -1." for n in names
    )
    credit_names = ", ".join(f"{n.lower()}_credit" for n in names)
    task_names = ", ".join(f"task_{n.lower()}" for n in names)
    template = "\n".join(
        [f"{n.lower()}_credit = np.array([...])  # {num_timesteps} numbers" for n in names]
        + [f"task_{n.lower()} = <one integer 0..{n_actions - 1}>" for n in names]
    )
    return (
        f"You are a credit and task assignment critic taking in observations and actions of these "
        f"{len(names)} robots named {name_list}, together with the global reward from the world. "
        f"You must (1) assign a reward to each robot depending on how much its actions contributed "
        f"toward reaching the goal, and (2) recommend the next action each robot should take. "
        f"{credit_sentences} {task_sentences} Try to make sure your reward values for each robot "
        f"add up to less than ten.\n\n"
        f"Make sure each credit array is of size {num_timesteps} (one value per timestep) and each "
        f"task is a single integer. Make the variables named exactly {credit_names}, "
        f"{task_names}.\n\n"
        f"Write at most a few sentences of explanation. Your answer MUST end with exactly one "
        f"python code block of this form and nothing after it:\n"
        f"```python\n"
        f"import numpy as np\n"
        f"{template}\n"
        f"```"
    )


def build_base_prompt(env, traj, num_timesteps):
    """p^T_base = (p_env, p^T_desc, p_defn, p^T_task). Reuses the env's describe() and MCA's
    definitions; the task-assignment request lives in build_task_prompt above."""
    described = env.describe(traj)
    return "\n\n".join([
        described["env"],
        described["desc"],
        build_defn_prompt(env),
        build_task_prompt(env.agent_names, num_timesteps, env.n_actions),
    ])


def build_batch_task_prompt(agent_names, episode_lengths, n_actions):
    """Batch p^T_task: score SEVERAL episodes at once (so credit is calibrated by comparison) AND
    recommend an action per robot per episode. Env-agnostic -- relies on the definitions above."""
    names = list(agent_names)
    name_list = " and ".join([", ".join(names[:-1]), names[-1]]) if len(names) > 1 else names[0]
    n_ep = len(episode_lengths)
    credit_tmpl = "\n".join(
        f"{n.lower()}_credit_{e} = np.array([...])  # episode {e}, exactly {T} numbers"
        for e, T in enumerate(episode_lengths, 1) for n in names
    )
    task_tmpl = "\n".join(f"task_{n.lower()} = <one integer 0..{n_actions - 1}>" for n in names)
    all_credit = ", ".join(f"{n.lower()}_credit_{e}" for e in range(1, n_ep + 1) for n in names)
    all_task = ", ".join(f"task_{n.lower()}" for n in names)
    return (
        f"You are a credit and task assignment critic. You are given {n_ep} SEPARATE episodes of "
        f"the same environment played by the robots {name_list}, with the global reward at every "
        f"timestep. You must (1) assign a per-timestep credit to each robot in EACH episode, and "
        f"(2) recommend, for each robot, the single best action (an integer 0..{n_actions - 1}) it "
        f"should take next time to earn the most reward.\n\n"
        f"For the credit, follow the temporal and structural credit-assignment definitions above: "
        f"give more credit to the actions that led toward the goal (that timestep or an earlier "
        f"one), and credit each robot for its own contribution rather than the team's. Because the "
        f"environment reward is sparse, DO NOT wait for a reward to appear: adjust the sparse "
        f"reward by implicitly forming sub-goals (e.g. moving toward the target, positioning to "
        f"collaborate) and giving a robot intermediary credit whenever it makes progress toward the "
        f"goal, even in a timestep or an episode where no reward was earned. Try to keep each "
        f"robot's per-episode credit adding up to less than ten.\n\n"
        f"For the task, infer from the episodes which action leads to the best team outcome and "
        f"recommend it to each robot (task_{names[0].lower()}, etc.).\n\n"
        f"Work through your reasoning explicitly before answering, in this structure:\n"
        f"1. **Temporal Credit Assignment**: where did the reward appear in each episode, and which "
        f"earlier actions led toward it?\n"
        f"2. **Structural Credit Assignment**: which robot actually contributed, and which "
        f"freeloaded?\n"
        f"3. **Collaboration**: did the robots need to cooperate? Note under- or "
        f"over-collaboration.\n\n"
        f"Then give ONE python code block naming exactly these variables -- credit arrays "
        f"({all_credit}) and task integers ({all_task}):\n"
        f"```python\n"
        f"import numpy as np\n"
        f"{credit_tmpl}\n"
        f"{task_tmpl}\n"
        f"```"
    )


def build_batch_prompt(env, trajs):
    """Full batch TACA prompt: shared env/definitions once, then all episodes, then the task."""
    described = env.describe()
    lengths = [len(t) for t in trajs]
    blocks = "\n\n".join(
        f"Episode {e} has {len(t)} timesteps:\n{env.serialize(t)}" for e, t in enumerate(trajs, 1)
    )
    base = "\n\n".join([
        described["env"], described["desc"], build_defn_prompt(env),
        build_batch_task_prompt(env.agent_names, lengths, env.n_actions),
    ])
    return (
        f"{base}\n\nHere are the {len(trajs)} episodes:\n{blocks}\n\n"
        f"Remember: output only the credit arrays and task integers named above.\nYour answer:"
    )
