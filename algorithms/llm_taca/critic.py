"""LLM-TACA critic: same model/generation as the MCA critic, but with the task-assignment
prompt and the F_TACA parser (credit + per-agent target)."""
from ..llm_mca.critic import LLMCritic
from .prompts import build_base_prompt, build_critic_prompt, build_batch_prompt
from .parser import parse_credits_and_tasks, parse_batch


class LLMTACACritic(LLMCritic):
    def assign(self, traj, env, trajectory_text):
        """Returns (credits [n_agents, T], recommended_actions [n_agents], raw_text)."""
        T = len(traj)
        base = build_base_prompt(env, traj, T)
        prompt = build_critic_prompt(base, trajectory_text)
        text = self.generate(prompt)
        credits, actions = parse_credits_and_tasks(
            text, env.n_agents, T, env.agent_names, env.n_actions)
        return credits, actions, text

    def _assign_group(self, trajs, env):
        prompt = build_batch_prompt(env, trajs)
        text = self.generate(prompt)
        credits, actions = parse_batch(
            text, env.n_agents, [len(t) for t in trajs], env.agent_names, env.n_actions)
        return credits, actions, text

    def assign_batch(self, trajs, env):
        """Score the batch (credit + per-agent task recommendation). With group_size>0, split into
        groups of that many episodes (one LLM call each) so the prompt fits the context window;
        the task recommendation from later groups overrides earlier ones.
        Returns (list of credit matrices, per-agent recommended actions, raw_text)."""
        g = self.group_size
        if not g or g >= len(trajs):
            return self._assign_group(trajs, env)
        all_credits, actions, texts = [], [None] * env.n_agents, []
        for s in range(0, len(trajs), g):
            credits, acts, text = self._assign_group(trajs[s:s + g], env)
            all_credits.extend(credits)
            actions = [acts[i] if acts[i] is not None else actions[i] for i in range(env.n_agents)]
            texts.append(text)
        return all_credits, actions, "\n---\n".join(texts)
