
import os
import re
import json
import time
from datetime import datetime
os.environ["GEMINI_API_KEY"] = ""
from tools import TOOLS, render_tool_descriptions

PROVIDERS = {
    "gemini": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "model": "gemini-3.5-flash-lite",
        "key_env": "GEMINI_API_KEY",
    },
    "deepseek": {
        "base_url": "https://api.deepseek.com/v1",
        "model": "deepseek-chat",
        "key_env": "DEEPSEEK_API_KEY",
    },
    "ollama": {
        "base_url": "http://localhost:11434/v1",
        "model": "qwen2.5:7b",
        "key_env": None,
    },
}

PROVIDER = "gemini"

class LLMClient:

    MIN_INTERVAL = 4.0
    MAX_RETRY = 5

    _last_call = 0.0

    def __init__(self, provider=PROVIDER):
        from openai import OpenAI

        cfg = PROVIDERS[provider]
        key = os.environ.get(cfg["key_env"], "") if cfg["key_env"] else "ollama"
        if not key:
            raise RuntimeError(
                f"没找到环境变量 {cfg['key_env']}。\n"
                f"在 PyCharm 里 Run -> Edit Configurations -> Environment variables 加上，\n"
                f"或临时在代码里写 os.environ['{cfg['key_env']}'] = 'xxx'"
            )
        self.client = OpenAI(api_key=key, base_url=cfg["base_url"])
        self.model = cfg["model"]

    def __call__(self, prompt: str, stop=None) -> str:
        from openai import RateLimitError
        wait = self.MIN_INTERVAL - (time.time() - LLMClient._last_call)
        if wait > 0:
            time.sleep(wait)

        delay = 10.0
        for attempt in range(1, self.MAX_RETRY + 1):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    stop=stop,
                    max_tokens=512,
                )
                LLMClient._last_call = time.time()
                return resp.choices[0].message.content or ""
            except RateLimitError:
                if attempt == self.MAX_RETRY:
                    raise
                print(f"  [429] 触发限流，{delay:.0f}s 后重试 ({attempt}/{self.MAX_RETRY})")
                time.sleep(delay)
                delay *= 2      # 指数退避
        return ""


class MockLLM:

    def __init__(self, script):
        self.script = list(script)
        self.i = 0

    def __call__(self, prompt: str, stop=None) -> str:
        if self.i >= len(self.script):
            return "Final Answer: 脚本结束"
        out = self.script[self.i]
        self.i += 1
        return out

SYSTEM_TEMPLATE = """你是一个医学问答助手。你可以调用以下工具来回答问题：

{tool_descriptions}

严格按照以下格式回复，一次只输出一个步骤：

Thought: 你现在的思考
Action: 要调用的工具名，必须是 [{tool_names}] 之一
Action Input: 传给工具的参数

系统会返回：
Observation: 工具的执行结果

这个循环可以重复多次。当你有足够信息时，输出：

Thought: 我已经可以回答了
Final Answer: 你的最终答案

规则：
- 判断医学问题的 yes/no/maybe 结论前，必须先用 search_literature 检索证据
- 调用 classify_answer 时，参数格式是：问题 || 证据文本
- 如果 classify_answer 返回的三类概率都很接近（都低于 0.5），说明模型没把握，
  应该换个关键词再检索一次，而不是直接下结论
- 不要自己编造 Observation，Observation 只能由系统返回

现在开始。

Question: {question}
"""


ACTION_RE = re.compile(r"Action\s*:\s*(.+?)\s*[\n\r]+\s*Action\s*Input\s*:\s*(.+)", re.S)
FINAL_RE = re.compile(r"Final\s*Answer\s*:\s*(.+)", re.S)


class ParseError(Exception):
    pass


def parse_step(text: str):
    m = FINAL_RE.search(text)
    if m:
        return ("final", m.group(1).strip())

    m = ACTION_RE.search(text)
    if m:
        action = m.group(1).strip().strip("`\"' ")
        action_input = m.group(2).strip()
        for marker in ["\nObservation", "\nThought", "\nFinal Answer"]:
            if marker in action_input:
                action_input = action_input.split(marker)[0]
        return ("action", action, action_input.strip().strip("`\"' "))

    raise ParseError("既没解析到 Action/Action Input，也没解析到 Final Answer")


MAX_STEPS = 6
MAX_PARSE_RETRY = 2
OBS_MAX_CHARS = 1200
MAX_SAME_ACTION = 3
MAX_PUSHBACK = 2


class ReActAgent:
    def __init__(self, llm=None, verbose=True):
        self.llm = llm or LLMClient()
        self.verbose = verbose
        self.trajectory = []

    def _log(self, role, content):
        self.trajectory.append({"role": role, "content": content})
        if self.verbose:
            print(f"\033[36m{role}:\033[0m {content}")

    def run(self, question: str) -> str:
        self.trajectory = []
        tool_names = ", ".join(TOOLS.keys())
        prompt = SYSTEM_TEMPLATE.format(
            tool_descriptions=render_tool_descriptions(),
            tool_names=tool_names,
            question=question,
        )
        self._log("Question", question)

        parse_fails = 0
        action_counts = {}
        has_valid_evidence = False
        pushbacks = 0

        for step in range(1, MAX_STEPS + 1):
            raw = self.llm(prompt, stop=["Observation:"])

            try:
                parsed = parse_step(raw)
                parse_fails = 0
            except ParseError as e:
                parse_fails += 1
                self._log("ParseError", f"{e} | 原始输出: {raw[:200]}")
                if parse_fails > MAX_PARSE_RETRY:
                    return "[失败] 连续多次格式错误，已放弃"
                prompt += raw + "\nObservation: 输出格式不对，请严格按 Thought/Action/Action Input 或 Final Answer 的格式重写。\n"
                continue

            if parsed[0] == "final":
                answer = parsed[1]
                asserts_conclusion = bool(
                    re.match(r'^\s*(yes|no|是|否|不|No\b|Yes\b)', answer, re.I)
                )
                if asserts_conclusion and not has_valid_evidence:
                    pushbacks += 1
                    self._log("OutputGuard", f"结论缺乏工具证据支撑，打回第 {pushbacks} 次")
                    if pushbacks <= MAX_PUSHBACK:
                        prompt += raw + (
                            "\nObservation: [输出校验未通过] 你给出了明确结论，"
                            "但整个过程中没有任何一次 classify_answer 返回了可用的预测结果。"
                            "不允许使用你自己的医学知识作答，只能依据工具返回的证据。"
                            "请改为：补充检索后重新分类，或在 Final Answer 中明确说明"
                            "「证据不足，无法判断」。\n"
                        )
                        continue

                    answer = "[无工具证据支撑，以下为模型自述，不可采信] " + answer

                self._log("Final Answer", answer)
                return answer

            _, action, action_input = parsed
            self._log(f"Step {step} Action", f"{action}({action_input[:120]})")

            if action not in TOOLS:
                obs = f"没有名为 {action} 的工具，可用工具：{tool_names}"
            else:
                try:
                    obs = TOOLS[action]["func"](action_input)
                except Exception as e:
                    obs = f"工具执行出错: {type(e).__name__}: {e}"

            obs = str(obs)


            if action == "classify_answer" and obs.startswith("预测:"):
                has_valid_evidence = True

            if len(obs) > OBS_MAX_CHARS:
                obs = obs[:OBS_MAX_CHARS] + " ...(已截断)"


            action_counts[action] = action_counts.get(action, 0) + 1
            if action_counts[action] >= MAX_SAME_ACTION:
                obs += (
                    f"\n\n[系统强制提示] 你已经调用 {action} {action_counts[action]} 次了。"
                    f"不要再调用这个工具。请基于目前已有的信息，"
                    f"调用其他工具或直接输出 Final Answer。"
                )
                self._log("LoopGuard", f"{action} 已调用 {action_counts[action]} 次，注入强制提示")

            self._log("Observation", obs[:300])
            prompt += raw + f"\nObservation: {obs}\n"


        self._log("MaxSteps", "步数耗尽，要求模型基于现有信息作答")
        prompt += "\nThought: 步数已用完，我必须基于现有信息直接回答。\nFinal Answer:"
        try:
            tail = self.llm(prompt, stop=None)
            answer = tail.strip().split("\n")[0]
            self._log("Final Answer (forced)", answer)
            return answer
        except Exception:
            return "[失败] 超过最大步数仍未得出结论"

    def save_trajectory(self, path="trajectories.jsonl", note=""):

        rec = {
            "time": datetime.now().isoformat(timespec="seconds"),
            "note": note,
            "trajectory": self.trajectory,
        }
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def offline_test():
    print("=" * 60)
    print("离线测试（MockLLM，不调用真实 LLM）")
    print("=" * 60)

    script = [

        "Thought: 先找证据\nAction: search_literature\nAction Input: aerobic exercise cognitive function older adults",

        "Thought: 调分类\nAction: classify_it\nAction Input: xxx",

        "我觉得答案应该是 yes 吧",

        "Thought: 够了\nFinal Answer: yes，有多项随机试验支持",
    ]

    agent = ReActAgent(llm=MockLLM(script))
    result = agent.run("Does aerobic exercise improve cognitive function in older adults?")
    print(f"\n返回: {result}")
    agent.save_trajectory(note="offline_test")
    print("\n轨迹已写入 trajectories.jsonl")


def real_run(questions=None):
    agent = ReActAgent()
    if questions is None:
        from tools import sample_real_questions
        questions = sample_real_questions(n=1, seed=0)
        print(f"测试问题（取自语料）: {questions[0]}")
    for q in questions:
        print("\n" + "=" * 60)
        agent.run(q)
        agent.save_trajectory(note=q)
        time.sleep(5)



BATCH_QUESTIONS = [
    "Does aerobic exercise improve cognitive function in older adults?",
    "Is metformin associated with reduced cancer risk in diabetic patients?",
    "Does vitamin D supplementation prevent respiratory infections?",
    "如果一组风险从 30% 降到 24%，相对风险下降了多少百分比？",
]


if __name__ == "__main__":
    real_run()
