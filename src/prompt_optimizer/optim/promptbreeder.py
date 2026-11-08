"""PromptBreeder-style self-referential evolution (Fernando et al., 2023).

A population of *units*, each a task prompt (the instruction) paired with a
*mutation prompt* (the text that tells the model how to mutate it), evolves by
binary tournament: two units are drawn, the fitter one (accuracy on one fixed
training minibatch shared by every unit, so fitness values are comparable) is
copied over the weaker one, and the copy is mutated by one operator drawn
uniformly from the paper's classes:

* **Direct mutation.** Zero-order generation (a new prompt from the problem
  description and a thinking style, ignoring the parent) and first-order
  generation (the unit's mutation prompt applied to the parent).
* **Estimation of distribution.** The model sees the whole population and
  continues the list with a new prompt (EDA mutation), or sees the history of
  elites in chronological order (lineage mutation).
* **Hypermutation.** The mutation prompt itself is mutated (first-order:
  "improve this mutation prompt"), then applied: this is what makes the method
  self-referential.
* **Lamarckian mutation.** A correct working-out from the training set is
  turned back into an instruction ("working out to task prompt").
* **Crossover.** After mutation, with probability ``crossover_rate``, the task
  prompt is replaced by one from another unit drawn in proportion to fitness.

The thinking styles and mutation prompts below are short originals written
for this repository in the spirit of the paper's lists. The paper also evolves
few-shot contexts; here demonstrations are left to the dedicated optimizers.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from prompt_optimizer.optim.base import Optimizer, OptimizerContext, clip
from prompt_optimizer.program import PromptProgram

THINKING_STYLES: tuple[str, ...] = (
    "Break the problem into smaller steps and solve them in order.",
    "Check each intermediate result before using it.",
    "Restate the problem in your own words before solving it.",
    "Consider what a careful expert would notice first.",
    "Look for the quantity the question actually asks for.",
    "Work backwards from what is asked to what is given.",
    "List the facts given, then combine them.",
    "Watch for traps and details that are easy to misread.",
)
MUTATION_PROMPTS: tuple[str, ...] = (
    "Make this instruction more specific about how to reach a correct answer.",
    "Rewrite the instruction so a beginner could follow it.",
    "Shorten the instruction while keeping its most useful advice.",
    "Add one piece of advice that would prevent a common mistake.",
    "Rephrase the instruction as a short step-by-step procedure.",
    "Change the instruction in a way that makes the answer more reliable.",
)

_SYSTEM = (
    "You write and improve instructions for a language model. Reply with the requested "
    "instruction inside <instruction></instruction> tags."
)


@dataclass
class Unit:
    task_prompt: str
    mutation_prompt: str
    fitness: float = 0.0


@dataclass
class PromptBreeder(Optimizer):
    """Binary-tournament evolution of task prompts and mutation prompts."""

    population: int = 6
    generations: int = 6
    tournaments_per_generation: int = 3
    minibatch: int = 12
    crossover_rate: float = 0.1
    n_finalists: int = 2
    name: str = "promptbreeder"
    operators: tuple[str, ...] = field(
        default=(
            "zero_order",
            "first_order",
            "eda",
            "lineage",
            "hypermutation",
            "lamarckian",
        )
    )

    # -- operator prompts -------------------------------------------------------------------
    def zero_order(self, ctx: OptimizerContext, style: str) -> str:
        return (
            f"Problem description: {ctx.task.description}\n\nThinking style: {style}\n\n"
            "Write an instruction that would help a model solve problems of this kind."
        )

    def first_order(self, mutation_prompt: str, task_prompt: str) -> str:
        return f"{mutation_prompt}\n\nINSTRUCTION: {task_prompt}\n\nINSTRUCTION MUTANT:"

    def eda(self, prompts: list[str]) -> str:
        listing = "\n".join(f"{i}. {p}" for i, p in enumerate(prompts, start=1))
        return (
            "Here is a list of instructions for the same task, in random order:\n"
            f"{listing}\n\nContinue the list with one new instruction that is different "
            "from all of them."
        )

    def lineage(self, elites: list[str]) -> str:
        listing = "\n".join(f"{i}. {p}" for i, p in enumerate(elites, start=1))
        return (
            "Here is the history of the best instructions so far, from oldest to newest, each "
            f"better than the one before:\n{listing}\n\nWrite the next, even better instruction."
        )

    def hyper(self, mutation_prompt: str) -> str:
        return (
            "Improve the following prompt, which tells a model how to rewrite an instruction. "
            f"Make it lead to better rewrites.\n\nPROMPT: {mutation_prompt}"
        )

    def lamarckian(self, ctx: OptimizerContext, question: str, working: str) -> str:
        return (
            "I gave a friend an instruction and some advice. Here is a correct example of their "
            f"work:\n\nProblem:\n{clip(question, 600)}\n\nWorking out:\n{clip(working, 700)}"
            "\n\nWhat was the instruction? Write it so it would help with other problems of "
            f"this kind ({ctx.task.description})."
        )

    # -- evolution --------------------------------------------------------------------------
    def _fitness(self, ctx: OptimizerContext, seed: PromptProgram, units: list[Unit]) -> bool:
        """Score units not yet scored on the shared fitness minibatch (False if unaffordable)."""

        batch = self._batch
        pending = [u for u in units if u.task_prompt not in self._scores]
        if pending and not ctx.affordable(len(batch) * len(pending), self._reserve):
            return False
        pending = list({u.task_prompt: u for u in pending}.values())
        programs = [seed.with_instruction(u.task_prompt) for u in pending]
        for unit, result in zip(
            pending, ctx.score_batch_many(programs, batch, note="fitness"), strict=True
        ):
            self._scores[unit.task_prompt] = result.mean
        for unit in units:
            unit.fitness = self._scores[unit.task_prompt]
        return True

    def _mutate(self, ctx: OptimizerContext, units: list[Unit], parent: Unit) -> Unit:
        op = ctx.rng.choice(self.operators)
        mutation_prompt = parent.mutation_prompt
        if op == "zero_order":
            prompt = self.zero_order(ctx, ctx.rng.choice(THINKING_STYLES))
        elif op == "first_order":
            prompt = self.first_order(mutation_prompt, parent.task_prompt)
        elif op == "eda":
            others = [u.task_prompt for u in units]
            ctx.rng.shuffle(others)
            prompt = self.eda(list(dict.fromkeys(others)))
        elif op == "lineage":
            prompt = self.lineage(self._elites[-6:])
        elif op == "hypermutation":
            new_mp = ctx.propose_many(_SYSTEM, [self.hyper(mutation_prompt)], purpose="mutate")
            mutation_prompt = new_mp[0] or mutation_prompt
            prompt = self.first_order(mutation_prompt, parent.task_prompt)
        else:
            correct = [r for r in self._last_rollouts if r.score >= 1.0]
            if correct:
                pick = ctx.rng.choice(correct)
                prompt = self.lamarckian(ctx, pick.example.question, pick.output)
            else:
                prompt = self.first_order(mutation_prompt, parent.task_prompt)
        texts = ctx.propose_many(_SYSTEM, [prompt], purpose="mutate")
        task_prompt = texts[0] or parent.task_prompt
        if ctx.rng.random() < self.crossover_rate and len(units) > 1:
            weights = [u.fitness + 1e-3 for u in units]
            task_prompt = ctx.rng.choices(units, weights=weights, k=1)[0].task_prompt
        return Unit(task_prompt, mutation_prompt)

    def _search(self, ctx: OptimizerContext, seed_program: PromptProgram) -> None:
        self._reserve = self.n_finalists * ctx.val_cost
        self._scores: dict[str, float] = {}
        self._batch = ctx.sample(ctx.trainset, self.minibatch)
        seeds = [self.zero_order(ctx, ctx.rng.choice(THINKING_STYLES))]
        seeds += [
            self.first_order(ctx.rng.choice(MUTATION_PROMPTS), seed_program.instruction)
            for _ in range(self.population - 2)
        ]
        texts = ctx.propose_many(_SYSTEM, seeds, purpose="mutate")
        prompts = list(dict.fromkeys([seed_program.instruction, *[t for t in texts if t]]))
        units = [Unit(p, ctx.rng.choice(MUTATION_PROMPTS)) for p in prompts]
        if not self._fitness(ctx, seed_program, units):
            return
        self._elites = [max(units, key=lambda u: u.fitness).task_prompt]
        for _ in range(self.generations):
            ctx.step += 1
            best = max(units, key=lambda u: u.fitness)
            program = seed_program.with_instruction(best.task_prompt)
            self._last_rollouts = list(ctx.evaluator.evaluate(program, self._batch).rollouts)
            children: list[tuple[int, Unit]] = []
            for _ in range(min(self.tournaments_per_generation, len(units) // 2)):
                a, b = ctx.rng.sample(range(len(units)), 2)
                winner, loser = (a, b) if units[a].fitness >= units[b].fitness else (b, a)
                children.append((loser, self._mutate(ctx, units, units[winner])))
            trial = list(units)
            for index, child in children:
                trial[index] = child
            if not self._fitness(ctx, seed_program, trial):
                break
            units = trial
            best = max(units, key=lambda u: u.fitness)
            if best.task_prompt != self._elites[-1]:
                self._elites.append(best.task_prompt)
        ranked = sorted(units, key=lambda u: -u.fitness)
        finalists = list(dict.fromkeys(u.task_prompt for u in ranked))[: self.n_finalists]
        ctx.score_finalists([seed_program.with_instruction(p) for p in finalists])
