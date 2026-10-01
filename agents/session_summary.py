"""Bounded working-session restarts with an immutable global turn namespace.

Graph references keep absolute turn indices. Training exports each session as a
separate prompt/response pair, so log probabilities never cross a context reset.
"""

import copy
import re
import uuid
import random


SUMMARY_REQUEST = (
    "Summarize the work so far for continuation in a fresh context. Preserve the "
    "task, verified facts, file paths, changes, failed attempts and remaining work. "
    "Do not call tools or answer the task. Return only <summary>...</summary>."
)
CONTINUE_REQUEST = "Continue the task from the summary. Use the normal task tools."


class SessionSummaryMixin:
    def _session(self, turn_cut=None):
        for session in reversed(getattr(self, "summary_sessions", [])):
            if turn_cut is None or turn_cut >= session["start"]:
                return session
        return None

    def working_messages(self, turn_cut=None):
        end = len(self.chat) if turn_cut is None else turn_cut
        session = self._session(end)
        if session is None:
            return self.chat[:end]
        return session["prefix"].messages() + self.chat[session["start"]:end]

    def context(self, turn_cut=None):
        session = self._session(turn_cut)
        if session is None:
            return super().context(turn_cut)
        end = len(self.chat) if turn_cut is None else turn_cut
        return (sum(session["prefix"].chat_ids, [])
                + sum(self.chat_ids[session["start"]:end], [])
                + self.get_generation_prompt())

    def context_ids(self, messages=None):
        return self.context()

    def get_turn_context(self, i):
        if self._session(i) is None:
            return super().get_turn_context(i)
        tokens = self._render_prefix(self.working_messages(i + 1))
        previous = self._render_prefix(self.working_messages(i))
        if tokens[:len(previous)] == previous:
            return tokens[len(previous):]
        suffix = self._render_prefix([self.chat[i]])
        if not suffix or tokens[-len(suffix):] != suffix:
            raise ValueError("Summary session template changed an unisolatable turn")
        return suffix

    def response_tokens_used(self):
        # Includes observations and summary generation, just like the original
        # response budget. Restarting never grants a fresh response allowance.
        return max(len(sum(self.chat_ids, [])) - self.prompt_ids_len, 0)

    def remaining_generation_tokens(self):
        return max(self.config.response_length - self.response_tokens_used()
                   - len(self.get_generation_prompt()), 0)

    async def maybe_restart_session(self, reserve_tokens=0):
        plugin = self.config.plugin
        if not getattr(plugin, "enable_summary", False):
            return False
        sessions = getattr(self, "summary_sessions", [])
        if len(sessions) >= int(getattr(plugin, "summary_max_restarts", 4)):
            return False
        window = self.config.prompt_length + self.config.response_length
        threshold = int(getattr(plugin, "summary_context_threshold", 0) or window // 2)
        if len(self.context()) < threshold:
            return False
        # Retry only after another meaningful chunk of work, not every turn
        # when a model produces an invalid or overlarge summary.
        if len(self.chat) < getattr(self, "_summary_retry_turn", 0):
            return False
        self._summary_retry_turn = len(self.chat) + 4
        before = len(self.context())
        self.append({"role": "user", "content": SUMMARY_REQUEST})
        continuation = {"role": "user", "content": CONTINUE_REQUEST}
        continuation_cost = len(self._render_prefix([continuation]))
        available = self.remaining_generation_tokens() - int(reserve_tokens) - continuation_cost
        cap = min(int(getattr(plugin, "summary_max_tokens", 512)), available)
        if cap < 32:
            self.rollback()
            return False
        response = await self.step(max_new_tokens=cap)
        if response is None:
            self.rollback()
            return False
        visible = response.rsplit("</think>", 1)[-1].strip()
        match = re.fullmatch(r"<summary>(.*?)</summary>", visible, re.S)
        if not match or not match.group(1).strip():
            self.append({"role": "user", "content": "Continue the task using the available evidence."})
            return False
        from .utils import AgentContext

        prompt = copy.deepcopy(self.chat[:self.prompt_turn])
        prompt[-1]["content"] += "\n\nPrevious session summary (working notes):\n" + match.group(1).strip()
        # Do not let truncate_prompt silently drop the original task or summary.
        if len(self._render_prefix(prompt)) > self.prompt_length:
            self.append({"role": "user", "content": "Continue the task using the available evidence."})
            return False
        prefix = AgentContext(prompt, self.tokenizer, self.config,
                              prompt_turn=len(prompt),
                              chat_template_kwargs=self.chat_template_kwargs)
        if len(prefix.context()) + continuation_cost >= before:
            self.append({"role": "user", "content": "Continue the task using the available evidence."})
            return False
        sessions.append({"start": len(self.chat), "prefix": prefix})
        self.summary_sessions = sessions
        self.context_uid = str(uuid.uuid4())
        self.append(continuation)
        print(f"[SESSION SUMMARY] restart={len(sessions)} working_tokens={len(self.context())} remaining={self.remaining_generation_tokens()}")
        return True

    def session_views(self):
        """Build views without modifying the archive or its turn numbering."""
        sessions = getattr(self, "summary_sessions", [])
        if not sessions:
            yield 0, 0, self
            return
        boundaries = [0] + [s["start"] for s in sessions] + [len(self.chat)]
        arrays = ("chat", "chat_ids", "chat_completions", "log_probs", "token_mask", "additional_info")
        for index, (start, end) in enumerate(zip(boundaries, boundaries[1:])):
            view = copy.copy(self)
            prefix = sessions[index - 1]["prefix"] if index else None
            for attr in arrays:
                setattr(view, attr, (list(getattr(prefix, attr)) if prefix else [])
                        + list(getattr(self, attr)[start:end]))
            view.prompt_turn = prefix.prompt_turn if prefix else self.prompt_turn
            view.summary_sessions = []
            yield index, start, view

    async def session_data(self):
        """Yield token-aligned training segments, retaining absolute turn maps."""
        from .utils import AgentContext

        for index, start, view in self.session_views():
            data = await AgentContext.get_data(view)
            data["response_turn_token_indices"] = {
                turn - (view.prompt_turn if index else 0) + start: ids
                for turn, ids in data["response_turn_token_indices"].items()
            }
            data["summary_session_index"] = index
            yield data


async def iter_agent_data(agents, is_train, max_traj=None):
    names = list(agents) if is_train else ["main"]
    if max_traj is not None and len(names) > max_traj:
        selected = [0] + sorted(random.sample(range(1, len(names)), k=max_traj - 1))
        names = [names[i] for i in selected]
    for name in names:
        async for data in agents[name].session_data():
            if is_train:
                yield name, data
        if not is_train:
            # Evaluation keeps the full audit transcript and exports the last
            # working session's tokens; it is never fed into a policy update.
            data["messages"] = agents[name].messages()
            yield name, data
