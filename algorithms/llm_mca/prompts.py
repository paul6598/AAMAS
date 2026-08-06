"""Base prompt construction for the LLM critic, following the paper (Fig. 3):
p_base = (p_env, p_desc, p_defn, p_task).

The environment-specific pieces (p_env, p_desc, definition examples) come from the environment
(describe(traj) / get_example(kind)); this module owns the generic definitions (p_defn) and the
critic's task query (p_task). Definition and task-query wording follows the paper's published
prompt text (Fig. 3 and Definition Prompt 1) as closely as possible, adapted to the batch
setting where a whole episode is scored at once.
"""


def build_defn_prompt(env):
    """p_defn: the five definitions, each followed by a concrete example episode from the
    environment (the paper's utils.get_example)."""
    return "\n\n".join([
        "Definition of temporal credit assignment problem: When an environment has sparse "
        "rewards, an agent or group of agents often have to perform multiple actions before "
        "receiving a reward, making it difficult to associate the correct actions to the reward "
        "received. As a result, we aim to assign more reward to actions which led to rewards in "
        "the future.\n"
        "Here is an example of an episode where temporal credit assignment would have helped:\n"
        f"{env.get_example('temporal')}",

        "Definition of structural credit assignment problem: When maximizing a shared reward "
        "earned by the whole team, we must determine the individual contribution of each agent "
        "to that shared reward, so that each agent is rewarded according to its own actions "
        "rather than the team's.\n"
        "Here is an example:\n"
        f"{env.get_example('structural')}",

        "Definition of under-collaboration problem: Within these sparse-reward cooperative "
        "settings, a group of decentralized agents suffers from under-collaboration when fewer "
        "agents coordinate to attempt a sub-task than the number of agents the sub-task "
        "requires, causing it to fail.\n"
        "Here is an example:\n"
        f"{env.get_example('under-collaboration')}",

        "Definition of over-collaboration problem: Given a number of homogeneous agents, a group "
        "suffers from over-collaboration when more agents than necessary coordinate on a "
        "sub-task, resulting in inefficiencies and agent conflicts.\n"
        "Here is an example:\n"
        f"{env.get_example('over-collaboration')}",

        # Definition Prompt 1, verbatim from the paper.
        "Definition of Agreement Problem: Given a set of observations and actions as performed "
        "by the agents in the environment, there will be times when the agents will individually "
        "accomplish goals, but occasionally, we will need them to collaborate. When two agents "
        "correctly agree to collaborate on a task that requires two agents, then they have found "
        "a valid solution to the \"Agreement Problem\".\n"
        "Here is an example:\n"
        f"{env.get_example('agreement')}",
    ])


def build_task_prompt(agent_names, num_timesteps):
    """p_task: the critic's role and required output format, following the paper's Fig. 3 task
    query as closely as possible -- a MINIMAL request (assign credit by contribution, output the
    named arrays, keep each robot's total under ten). Temporal/structural spreading is left to the
    LLM guided by the definitions above, not dictated with dense/directional rules here."""
    names = list(agent_names)
    name_list = " and ".join([", ".join(names[:-1]), names[-1]]) if len(names) > 1 else names[0]
    array_sentences = " ".join(
        f"Along with your explanation, create a numpy array called {n.lower()}_credit with "
        f"numerical credit values for {n}." for n in names
    )
    array_names = ", ".join(f"{n.lower()}_credit" for n in names)
    template = "\n".join(
        f"{n.lower()}_credit = np.array([...])  # exactly {num_timesteps} numbers" for n in names
    )
    return (
        f"You are a credit assignment critic tasked with taking in observations and actions of "
        f"these {len(names)} robots named {name_list}. Along with these, I will also provide you "
        f"with global reward information from the world. Then you are supposed to assign a reward "
        f"to each of these robots depending on how much you think their actions contributed "
        f"toward reaching the goal. {array_sentences} Try to make sure your reward values for "
        f"each robot adds up to less than ten.\n\n"
        f"Make sure the numpy arrays you create are of size {num_timesteps} (one value per "
        f"timestep of the episode) and you make one for each of {name_list}, named exactly "
        f"{array_names}. Output the credit arrays themselves, not the observations or actions.\n\n"
        f"Write a short explanation, then end with exactly one python code block of this form:\n"
        f"```python\n"
        f"import numpy as np\n"
        f"{template}\n"
        f"```"
    )


def build_base_prompt(env, traj, num_timesteps, paper_faithful=False):
    """p_base = (p_env, p_desc, p_defn, p_task), per the paper's Fig. 3 layout."""
    described = env.describe(traj, include_progress=not paper_faithful)
    return "\n\n".join([
        described["env"],
        described["desc"],
        build_defn_prompt(env),
        build_task_prompt(env.agent_names, num_timesteps),
    ])


def build_critic_prompt(base_prompt, trajectory_text):
    return (
        f"{base_prompt}\n\nEpisode trajectory:\n{trajectory_text}\n\n"
        # Figure 3 asks for an explanation *with* the arrays.  Do not override that
        # requirement here: the prior "arrays only" suffix contradicted p_task and
        # made the single-trajectory path materially different from the paper prompt.
        f"Remember: explain the credit assignment, then provide the requested credit "
        f"numpy arrays (one per robot).\n"
        f"Your answer:"
    )


def build_batch_task_prompt(agent_names, episode_lengths, compact=False):
    """p_task for batch mode (paper Fig. 2): score SEVERAL episodes at once so the critic can
    compare them. Wording tracks the paper's task query ("credit toward reaching the goal") and its
    stated densification mechanism (Sec. 3.2: "adjust the sparse rewards ... by implicitly
    generating sub-tasks and rewarding sub-goals ... makes the reward signal denser"). We spell the
    sub-goal densification out explicitly because, unlike the paper's Gemma, Qwen does not densify
    on its own from the definitions alone (it returns ~0 credit on zero-reward episodes)."""
    names = list(agent_names)
    name_list = " and ".join([", ".join(names[:-1]), names[-1]]) if len(names) > 1 else names[0]
    n_ep = len(episode_lengths)
    template = "\n".join(
        f"{n.lower()}_credit_{e} = np.array([...])  # episode {e}, exactly {T} numbers"
        for e, T in enumerate(episode_lengths, 1) for n in names
    )
    all_names = ", ".join(
        f"{n.lower()}_credit_{e}" for e in range(1, n_ep + 1) for n in names
    )
    if compact:
        return (
            f"You are a credit assignment critic for robots {name_list}. Silently apply temporal "
            f"and structural credit assignment and check under-/over-collaboration. Give credit to "
            f"each robot's own actions that made progress toward the team goal, including useful "
            f"sub-goals before a sparse reward appears.\n\n"
            f"Score every transition from t to t+1 with this concrete checklist:\n"
            f"- Select a still-active feasible target apple. For an apple that one robot can load, "
            f"use that robot's Manhattan distance to an adjacent loading cell. For an apple that "
            f"requires combined levels, both robots must use distinct adjacent cells around the "
            f"SAME apple as their shared target.\n"
            f"- Give +1 when that robot's action reduces its distance to its assigned loading cell, "
            f"-1 when it increases the distance or repeatedly hits a boundary, and 0 when it makes "
            f"no useful change. Use the supplied progress_to_apples tuple instead of doing coordinate "
            f"arithmetic yourself. Positive progress toward the chosen active apple supports +1; "
            f"negative progress supports -1. Never give positive movement credit when every entry "
            f"in that robot's progress tuple is zero or negative. Do not reward a movement merely "
            f"because it is a movement.\n"
            f"- Give +2 to each necessary adjacent loader on a successful load. Give -1 for a load "
            f"that fails because the robot is not adjacent or because required partners did not "
            f"coordinate. Do not give one robot the other robot's credit.\n"
            f"- Once an apple disappears, stop assigning approach credit for it and evaluate the "
            f"next active target. Earlier actions on the successful approach remain positive "
            f"(temporal credit assignment).\n"
            f"Keep each robot's absolute per-episode credit total at or below ten. An all-zero "
            f"episode is INVALID: every episode must contain at least one nonzero credit. Even when "
            f"global reward is zero, apply the same transition checklist; concrete progress is "
            f"positive and counterproductive behavior is negative.\n\n"
            f"Output exactly one python code block and nothing else. Define exactly these arrays: "
            f"{all_names}. Every entry must be a numeric literal, and every array must have exactly "
            f"the stated number of entries. Do not use ellipses, prose, comments, or additional "
            f"variables.\n"
            f"```python\n"
            f"import numpy as np\n"
            f"{template}\n"
            f"```"
        )
    return (
        f"You are a credit assignment critic. You are given {n_ep} separate episodes of the same "
        f"environment played by the robots {name_list}, together with the global reward at every "
        f"timestep of each episode. For each episode, assign a per-timestep credit to each robot "
        f"depending on how much its actions contributed toward reaching the goal.\n\n"
        f"Follow the temporal and structural credit-assignment definitions above: give more credit "
        f"to the actions that led toward the goal (that timestep or an earlier one), and credit each "
        f"robot for its own contribution rather than the team's. Because the environment reward is "
        f"sparse, DO NOT wait for a reward to appear: adjust the sparse reward by implicitly forming "
        f"sub-goals (e.g. moving toward the target, positioning to collaborate) and giving a robot "
        f"intermediary credit whenever it makes progress toward the goal, even in a timestep or an "
        f"episode where no reward was earned. An all-zero episode is invalid: use positive credit "
        f"for concrete progress and negative credit for counterproductive or unhelpful actions, so "
        f"every episode has at least one nonzero value. Try to keep each robot's per-episode credit values "
        f"adding up to less than ten.\n\n"
        f"Work through your reasoning explicitly before answering, in this structure:\n"
        f"1. **Temporal Credit Assignment**: where did the reward appear in each episode, and which "
        f"earlier actions led to it? Assign more credit to the actions that led to the reward.\n"
        f"2. **Structural Credit Assignment**: which robot actually contributed? Credit each robot "
        f"for its own contribution and avoid rewarding a robot that freeloaded.\n"
        f"3. **Collaboration**: did the robots need to cooperate on an apple? Note any "
        f"under-collaboration (too few robots) or over-collaboration (more than needed).\n\n"
        f"Then give the credit arrays, with a short inline comment on each saying why, and finish "
        f"with a brief explanation of why each robot got what it got. Use one python code block "
        f"naming exactly these arrays ({all_names}), each of the stated size:\n"
        f"```python\n"
        f"import numpy as np\n"
        f"{template}\n"
        f"```"
    )


def build_batch_prompt(env, trajs, compact=False, paper_faithful=False):
    """Full batch prompt: shared env/definitions once, then all episodes, then the batch task."""
    described = env.describe(include_progress=not paper_faithful)  # generic (no per-episode start)
    lengths = [len(t) for t in trajs]
    blocks = "\n\n".join(
        f"Episode {e} has {len(t)} timesteps:\n"
        f"{env.serialize(t, include_progress=not paper_faithful)}" for e, t in enumerate(trajs, 1)
    )
    base = "\n\n".join([
        described["env"],
        described["desc"],
        build_defn_prompt(env),
        build_batch_task_prompt(env.agent_names, lengths, compact=compact),
    ])
    ending = (
        "Return exactly the requested python code block and nothing else."
        if compact else
        "Reason through the three points above first, then give the code block with the credit "
        "arrays named above."
    )
    return f"{base}\n\nHere are the {len(trajs)} episodes:\n{blocks}\n\n{ending}\nYour answer:"
