from kvagent.prompt_compiler import AgentTask, PromptCompiler, PromptMode


def task(name: str) -> AgentTask:
    return AgentTask(name=name, role=f"role-{name}", instruction=f"do-{name}")


def test_shared_prefix_is_identical_across_agents() -> None:
    compiler = PromptCompiler()
    left = compiler.compile(
        artifact="int main(void) { return 0; }",
        task=task("rust"),
        mode=PromptMode.SHARED_PREFIX,
        experiment_id="exp",
    )
    right = compiler.compile(
        artifact="int main(void) { return 0; }",
        task=task("security"),
        mode=PromptMode.SHARED_PREFIX,
        experiment_id="exp",
    )
    assert left.shared_prefix == right.shared_prefix
    assert left.shared_prefix_sha256 == right.shared_prefix_sha256
    assert left.text.startswith(left.shared_prefix)
    assert right.text.startswith(right.shared_prefix)


def test_naive_prompts_diverge_before_artifact() -> None:
    compiler = PromptCompiler()
    left = compiler.compile(
        artifact="CODE",
        task=task("rust"),
        mode=PromptMode.NAIVE,
        experiment_id="exp",
    )
    right = compiler.compile(
        artifact="CODE",
        task=task("security"),
        mode=PromptMode.NAIVE,
        experiment_id="exp",
    )
    assert left.text != right.text
    assert left.shared_prefix == ""
    assert right.shared_prefix == ""
